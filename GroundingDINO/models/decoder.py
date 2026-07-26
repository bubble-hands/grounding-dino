import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryInitializer(nn.Module):
    def __init__(self, num_queries=900, hidden_dim=256, text_dim=768):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.sim_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, text_features):
        queries = self.query_embed.weight.unsqueeze(0).repeat(text_features.size(0), 1, 1)

        text_cls = text_features[:, 0, :]
        text_proj = self.text_proj(text_cls).unsqueeze(1)
        query_proj = self.sim_proj(queries)

        sim = torch.matmul(query_proj, text_proj.transpose(1, 2)).squeeze(-1)
        top_k = torch.topk(sim, self.num_queries, dim=1).indices

        return torch.gather(queries, 1, top_k.unsqueeze(-1).expand(-1, -1, queries.size(-1)))


class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim=256, n_heads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.image_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.text_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout)
        )

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, query, image_features, text_features):
        q = query
        q = q + self.drop1(self.self_attn(q, q, q)[0])
        q = self.norm1(q)

        q = q + self.drop2(self.image_attn(q, image_features, image_features)[0])
        q = self.norm2(q)

        q = q + self.drop3(self.text_attn(q, text_features, text_features)[0])
        q = self.norm3(q)

        q = q + self.ffn(q)
        return q


class GroundingDINOTransformerDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.query_init = QueryInitializer(
            num_queries=cfg.MODEL.NUM_QUERIES,
            hidden_dim=cfg.MODEL.HIDDEN_DIM,
            text_dim=cfg.MODEL.TEXT_ENCODER.DIM
        )

        self.layers = nn.ModuleList([
            DecoderLayer(
                hidden_dim=cfg.MODEL.HIDDEN_DIM,
                n_heads=cfg.MODEL.DECODER.NUM_HEADS,
                dim_feedforward=cfg.MODEL.DECODER.DIM_FEEDFORWARD,
                dropout=0.1
            )
            for _ in range(cfg.MODEL.DECODER.NUM_LAYERS)
        ])

        self.text_proj = nn.Linear(cfg.MODEL.TEXT_ENCODER.DIM, cfg.MODEL.HIDDEN_DIM)

    def forward(self, image_features, text_features):
        queries = self.query_init(text_features)
        text_proj = self.text_proj(text_features)

        image_flatten = []
        for feat in image_features:
            B, C, H, W = feat.shape
            image_flatten.append(feat.flatten(2).transpose(1, 2))
        image_flatten = torch.cat(image_flatten, dim=1)

        for layer in self.layers:
            queries = layer(queries, image_flatten, text_proj)

        return queries