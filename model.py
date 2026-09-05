import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops

_MASK_VALUE = -1e9
_EPSILON = 1e-8


def scatter_softmax(scores, index, num_nodes):
    B = scores.shape[0]
    device = scores.device
    max_vals = torch.full((B, num_nodes), fill_value=_MASK_VALUE, device=device)
    max_vals.scatter_reduce_(1, index, scores, reduce='amax', include_self=True)
    max_gathered = torch.gather(max_vals, 1, index)
    exp_scores = torch.exp(scores - max_gathered)
    
    sum_exp = torch.zeros(B, num_nodes, device=device)
    sum_exp.scatter_add_(1, index, exp_scores)
    sum_gathered = torch.gather(sum_exp, 1, index)
    return exp_scores / (sum_gathered + _EPSILON)


class EdgeDenoiser(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, h_r, r_query_expand, conf_embeds, edge_mask):
        gate_input = torch.cat([h_r, r_query_expand, conf_embeds], dim=-1)
        relevance = self.gate_net(gate_input)
        relevance = relevance * edge_mask.unsqueeze(-1).float()
        return relevance


class ConfidenceEncoder(nn.Module):
    def __init__(self, d_model, scale=10.0):
        super().__init__()
        self.d_model = d_model
        self.register_buffer('B', torch.randn(1, d_model // 2) * scale)
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, scores, mask=None):
        if scores.dim() == 2:
            scores = scores.unsqueeze(-1) 
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            scores = scores.masked_fill(~mask, 0.0)
        x_proj = 2 * math.pi * scores @ self.B 
        x_features = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        return self.projection(x_features)


class GlobalLinearAttention(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_local, x_global):
        B, N_sub, D = x_local.shape
        N_all, _ = x_global.shape

        q = self.W_q(x_local)   
        k = self.W_k(x_global)  
        v = self.W_v(x_global)  

        q = einops.rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = einops.rearrange(k, 'n (h d) -> h n d', h=self.num_heads)
        v = einops.rearrange(v, 'n (h d) -> h n d', h=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        kvs = einops.einsum(k, v, 'h n d, h n D -> h d D')
        numerator = einops.einsum(q, kvs, 'b h n d, h d D -> b h n D')
        v_sum = einops.reduce(v, 'h n d -> h 1 d', 'sum') 
        v_sum = einops.rearrange(v_sum, 'h 1 D -> 1 h 1 D')
        numerator = numerator + v_sum

        k_sum = einops.reduce(k, 'h n d -> h d', 'sum') 
        denominator = einops.einsum(q, k_sum, 'b h n d, h d -> b h n')
        denominator = denominator + N_all
        denominator = einops.rearrange(denominator, 'b h n -> b h n 1')

        out = numerator / denominator
        out = einops.rearrange(out, 'b h n d -> b n (h d)')

        return self.norm(x_local + out)


class LogicReasoningEncoder(nn.Module):
    def __init__(self, n_rels, d_model, n_layers=3, tau=0.1, disable_former=False):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.tau = tau
        self.disable_former = disable_former

        self.beta_net = nn.Linear(d_model, 1)
        self.msg_layers = nn.ModuleList([nn.Linear(d_model * 5, d_model) for _ in range(n_layers)])
        self.update_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model)
        self.att_layers = nn.ModuleList([nn.Linear(d_model * 3, 1) for _ in range(n_layers)])
        self.edge_denoiser = EdgeDenoiser(d_model)

        if not disable_former:
            self.gla_layers = nn.ModuleList([GlobalLinearAttention(d_model) for _ in range(n_layers)])

    def forward(self, batch_data, r_query_embed, rel_embed_layer, conf_embeds, global_ent_embeds):
        edge_index = batch_data['edge_index']
        edge_type = batch_data['rels']
        edge_scores_raw = batch_data['scores'].unsqueeze(-1)
        edge_conf_mask = batch_data['edge_conf_mask']
        edge_mask = batch_data['edge_mask'].unsqueeze(-1).float()

        B, Max_N = batch_data['mask'].shape
        _, _, Max_E = edge_index.shape
        device = r_query_embed.device

        h_init = torch.zeros(B, Max_N, self.d_model).to(device)
        h_init[:, 0, :] = 1.0
        h = h_init.clone()

        h_r_static = rel_embed_layer(edge_type)                               
        r_query_expand = r_query_embed.unsqueeze(1).expand(-1, Max_E, -1)     

        # 始终计算去噪掩码
        edge_mask_1d = batch_data['edge_mask']
        denoise_mask = self.edge_denoiser(h_r_static, r_query_expand, conf_embeds, edge_mask_1d)
        target_indices_1d = edge_index[:, 1, :] 

        for k in range(self.n_layers):
            src_indices = edge_index[:, 0, :].unsqueeze(-1).expand(-1, -1, self.d_model)
            h_src = torch.gather(h, 1, src_indices)
            h_init_src = torch.gather(h_init, 1, src_indices)

            beta = torch.sigmoid(self.beta_net(h_r_static + r_query_expand))
            gate_known = torch.sigmoid((edge_scores_raw - beta) / self.tau)
            gate = torch.where(edge_conf_mask.unsqueeze(-1), gate_known,
                               torch.full_like(gate_known, 0.5))

            comp_feat = h_src * h_r_static
            msg_in = torch.cat([comp_feat, h_src, h_init_src, h_r_static, conf_embeds], dim=-1)
            raw_msg = F.relu(self.msg_layers[k](msg_in))

            # 始终计算关系感知注意力
            att_input = torch.cat([raw_msg, h_r_static, r_query_expand], dim=-1)
            att_score = F.leaky_relu(self.att_layers[k](att_input)).squeeze(-1) 
            att_score = att_score.masked_fill(batch_data['edge_mask'] == 0, _MASK_VALUE)
            alpha = scatter_softmax(att_score, target_indices_1d, Max_N).unsqueeze(-1) 

            weighted_msg = gate * alpha * raw_msg * edge_mask * denoise_mask

            target_indices = target_indices_1d.unsqueeze(-1).expand(-1, -1, self.d_model)
            aggr_out = torch.zeros_like(h)
            aggr_out.scatter_add_(1, target_indices, weighted_msg)

            h = h + self.update_layers[k](aggr_out)
            h = self.layer_norm(h)

            if not self.disable_former:
                h = self.gla_layers[k](h, global_ent_embeds)

        return h


class StructureFeatureEncoder(nn.Module):
    def __init__(self, n_rels, d_model, n_layers=3, disable_former=False):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.disable_former = disable_former

        self.dist_embed = nn.Embedding(10, d_model)
        self.msg_layers = nn.ModuleList([nn.Linear(d_model * 5, d_model) for _ in range(n_layers)])
        self.update_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.att_layers = nn.ModuleList([nn.Linear(d_model * 3, 1) for _ in range(n_layers)])
        self.jk_att = nn.Linear(d_model, 1)
        self.edge_denoiser = EdgeDenoiser(d_model)

        if not disable_former:
            self.gla_layers = nn.ModuleList([GlobalLinearAttention(d_model) for _ in range(n_layers)])

    def forward(self, batch_data, r_query_embed, rel_embed_layer, conf_embeds, global_ent_embeds):
        dists = batch_data['dists']
        edge_index = batch_data['edge_index']
        edge_type = batch_data['rels']
        node_mask = batch_data['mask']
        edge_mask = batch_data['edge_mask'].unsqueeze(-1).float()

        B, Max_N = dists.shape
        _, _, Max_E = edge_index.shape
        device = dists.device

        dist_emb = self.dist_embed(torch.clamp(dists, 0, 9))
        noise = torch.randn_like(dist_emb) * 0.1
        h = dist_emb + noise

        h_r_static = rel_embed_layer(edge_type)                               
        r_query_expand = r_query_embed.unsqueeze(1).expand(-1, Max_E, -1) 

        edge_mask_1d = batch_data['edge_mask']
        denoise_mask = self.edge_denoiser(h_r_static, r_query_expand, conf_embeds, edge_mask_1d)
        target_indices_1d = edge_index[:, 1, :] 
        layer_outputs = []

        for k in range(self.n_layers):
            src_indices = edge_index[:, 0, :].unsqueeze(-1).expand(-1, -1, self.d_model)
            h_src = torch.gather(h, 1, src_indices)
            dist_src = torch.gather(dist_emb, 1, src_indices)

            comp_feat = h_src * h_r_static
            msg_input = torch.cat([comp_feat, h_src, dist_src, h_r_static, conf_embeds], dim=-1)
            msg = F.relu(self.msg_layers[k](msg_input))

            att_input = torch.cat([msg, h_r_static, r_query_expand], dim=-1) 
            att_score = F.leaky_relu(self.att_layers[k](att_input)).squeeze(-1)
            att_score = att_score.masked_fill(batch_data['edge_mask'] == 0, _MASK_VALUE)
            alpha = scatter_softmax(att_score, target_indices_1d, Max_N).unsqueeze(-1)

            msg = alpha * msg * edge_mask * denoise_mask

            target_indices = target_indices_1d.unsqueeze(-1).expand(-1, -1, self.d_model)
            aggr_out = torch.zeros_like(h)
            aggr_out.scatter_add_(1, target_indices, msg)

            h = self.update_layers[k](aggr_out) + h
            
            if not self.disable_former:
                h = self.gla_layers[k](h, global_ent_embeds)
                
            layer_outputs.append(h)

        layer_stack = torch.stack(layer_outputs, dim=1)  
        jk_scores = self.jk_att(layer_stack).squeeze(-1) 
        jk_weights = F.softmax(jk_scores, dim=1)          
        h = (jk_weights.unsqueeze(-1) * layer_stack).sum(dim=1)  
        h = h * node_mask.unsqueeze(-1).float()

        return h


class KGReasoningModel(nn.Module):
    def __init__(self, n_ents, n_rels, d_model=64, n_layers=3, top_k_evd=3,
                 disable_sfe=False, disable_lre=False, disable_conf=False, disable_former=False,
                 conf_mask_prob=0.9):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.top_k_evd = top_k_evd

        self.disable_sfe = disable_sfe
        self.disable_lre = disable_lre
        self.disable_conf = disable_conf
        self.disable_former = disable_former
        self.conf_mask_prob = conf_mask_prob

        self.ent_embed = nn.Embedding(n_ents, d_model)
        self.rel_embed = nn.Embedding(n_rels, d_model)

        self.conf_encoder = ConfidenceEncoder(d_model)
        self.lre = LogicReasoningEncoder(n_rels, d_model, n_layers=n_layers, disable_former=disable_former)
        self.sfe = StructureFeatureEncoder(n_rels, d_model, n_layers=n_layers, disable_former=disable_former)

        self.W_p = nn.Linear(d_model, 1)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 4 + 1, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, 1)
        )

    def forward(self, h_idx, r_idx, t_idx, lre_data, sfe_data):
        r_query = self.rel_embed(r_idx)
        B = r_query.shape[0]
        device = r_query.device

        # 第一阶段：置信度编码与掩码 
        if self.disable_conf:
            lre_conf_emb = torch.zeros(B, lre_data['scores'].shape[1], self.d_model, device=device)
            sfe_conf_emb = torch.zeros(B, sfe_data['scores'].shape[1], self.d_model, device=device)
        else:
            lre_mask = lre_data['edge_conf_mask']
            sfe_mask = sfe_data['edge_conf_mask']
            if self.training and self.conf_mask_prob < 1.0:
                lre_rand = torch.rand(lre_data['scores'].shape + (1,), device=device)
                sfe_rand = torch.rand(sfe_data['scores'].shape + (1,), device=device)
                lre_mask = lre_mask.unsqueeze(-1) & (lre_rand < self.conf_mask_prob)
                sfe_mask = sfe_mask.unsqueeze(-1) & (sfe_rand < self.conf_mask_prob)
            lre_conf_emb = self.conf_encoder(lre_data['scores'], mask=lre_mask)
            sfe_conf_emb = self.conf_encoder(sfe_data['scores'], mask=sfe_mask)

        global_ent_embeds = self.ent_embed.weight

        # 第二阶段：解耦双流计算
        if self.disable_lre:
            H_ctx_global = torch.zeros(B, lre_data['mask'].shape[1], self.d_model, device=device)
        else:
            H_ctx_global = self.lre(lre_data, r_query, self.rel_embed, lre_conf_emb, global_ent_embeds)

        if self.disable_sfe:
            H_evd_global = torch.zeros(B, sfe_data['mask'].shape[1], self.d_model, device=device)
        else:
            H_evd_global = self.sfe(sfe_data, r_query, self.rel_embed, sfe_conf_emb, global_ent_embeds)

        # 第三阶段：特征提纯与表征提取
        D = self.d_model

        if self.disable_lre:
            z_exp = torch.zeros(B, D, device=device)
        else:
            node_mask = lre_data['mask']
            Max_N = H_ctx_global.size(1)
            rq_exp = r_query.unsqueeze(1).expand(-1, Max_N, -1)
            
            p_logits = self.W_p(H_ctx_global + rq_exp).squeeze(-1)
            p_logits = p_logits.masked_fill(~node_mask, _MASK_VALUE)
            alpha = F.softmax(p_logits, dim=1)

            curr_K = min(self.top_k_evd, Max_N)
            topk_vals, topk_idx = torch.topk(alpha, k=curr_K, dim=1)
            topk_vals = topk_vals / (topk_vals.sum(dim=1, keepdim=True) + 1e-8)
            
            topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
            topk_H = torch.gather(H_ctx_global, 1, topk_idx_exp)
            z_exp = (topk_H * topk_vals.unsqueeze(-1)).sum(dim=1)

        if self.disable_sfe:
            z_real = torch.zeros(B, D, device=device)
        else:
            z_real = H_evd_global[:, 0, :]

        # 第四阶段：特征交互与感知预测
        diff_feat = torch.abs(z_exp - z_real)
        mult_feat = z_exp * z_real
        cos_sim = F.cosine_similarity(z_exp, z_real, dim=-1).unsqueeze(-1)
        
        F_fuse = torch.cat([
            z_exp, z_real, diff_feat, mult_feat, cos_sim
        ], dim=-1)

        out = torch.sigmoid(self.classifier(F_fuse))
        out = torch.clamp(out.squeeze(-1), min=1e-5, max=1.0 - 1e-5)

        return out