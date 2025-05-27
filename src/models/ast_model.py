import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PatchEmbedding(nn.Module):
    def __init__(self, input_dims: Tuple[int, int] = (128, 173), patch_size: Tuple[int, int] = (16, 16), embed_dim: int = 768):
        super().__init__()
        self.input_dims = input_dims
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.num_patches_h = input_dims[0] // patch_size[0]
        self.num_patches_w = input_dims[1] // patch_size[1]
        self.num_patches = self.num_patches_h * self.num_patches_w

        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.input_dims[0] and W == self.input_dims[1], \
            f"Input dims {(H, W)} don't match expected {self.input_dims}"

        x = self.proj(x)

        x = x.flatten(2)

        x = x.transpose(1, 2)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, num_patches: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed[:, :N+1, :]

        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))

        x = x + self.mlp(self.norm2(x))

        return x


class ASTModel(nn.Module):
    def __init__(
        self,
        n_classes: int = 10,
        input_dims: Tuple[int, int] = (128, 173),
        patch_size: Tuple[int, int] = (16, 16),
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pretrained: Optional[str] = None
    ):
        super().__init__()

        self.n_classes = n_classes
        self.input_dims = input_dims
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(input_dims, patch_size, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = PositionalEncoding(num_patches, embed_dim, dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

        self._init_weights()

        if pretrained:
            self._load_pretrained(pretrained)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _load_pretrained(self, pretrained: str):
        if pretrained == "imagenet":
            print("Loading ImageNet pretrained weights (simulated)")
            pass
        else:
            print(f"Unknown pretrained option: {pretrained}")

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)

        x = self.pos_embed(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_token = x[:, 0]
        logits = self.head(cls_token)

        return logits


def create_ast_model(
    n_classes: int = 10,
    input_dims: Tuple[int, int] = (128, 173),
    patch_size: Tuple[int, int] = (16, 16),
    model_size: str = "base",
    dropout: float = 0.1,
    pretrained: Optional[str] = None
) -> ASTModel:
    
    configs = {
        "tiny": {"embed_dim": 192, "depth": 12, "num_heads": 3},
        "small": {"embed_dim": 384, "depth": 12, "num_heads": 6},
        "base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
        "large": {"embed_dim": 1024, "depth": 24, "num_heads": 16}
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")
    
    config = configs[model_size]
    
    return ASTModel(
        n_classes=n_classes,
        input_dims=input_dims,
        patch_size=patch_size,
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        dropout=dropout,
        pretrained=pretrained
    )


class ASTTransformer(ASTModel):
    pass