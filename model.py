import torch
import torch.nn as nn


class StableLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(0.1))

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)
        q, k = q.softmax(dim=-1), k.softmax(dim=-2)
        context = torch.matmul(k.transpose(-1, -2), v)
        out = torch.matmul(q, context)
        return self.to_out(out.transpose(1, 2).reshape(b, n, -1))


class PointTransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = StableLinearAttention(dim, heads=heads)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(0.1), nn.Linear(dim * 4, dim)
        )
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class ProductionTransformer(nn.Module):
    """
    Transformer utilizing a Global Physics Token for context.
    """

    def __init__(self, geo_dim, phys_dim, output_dim, model_dim=128, num_layers=4, num_heads=4):
        super().__init__()

        # 1. Embedding for Point Cloud
        self.geo_embed = nn.Sequential(
            nn.Linear(geo_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU()
        )

        # 2. Embedding for Global Physics (The "Token")
        self.phys_embed = nn.Sequential(
            nn.Linear(phys_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU()
        )

        self.layers = nn.ModuleList([
            PointTransformerBlock(model_dim, num_heads) for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, output_dim)
        )

    def forward(self, x_geo, x_phys):
        # x_geo: [Batch, N, 3]
        # x_phys: [Batch, 7]

        # Embed points
        tokens = self.geo_embed(x_geo)  # [B, N, Dim]

        # Embed physics and add sequence dimension -> [B, 1, Dim]
        global_token = self.phys_embed(x_phys).unsqueeze(1)

        # Prepend global token to sequence: [B, N+1, Dim]
        x = torch.cat([global_token, tokens], dim=1)

        # Run Transformer
        for layer in self.layers:
            x = layer(x)

        # Cut off the global token (first element), return only point predictions
        # Output: [B, N, Out]
        return self.head(x[:, 1:, :])