import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, 
            embed_dim,
            kernel_size=patch_size,
            stride = patch_size
        )
    def forward(self,x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1,2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()

        assert dim % num_heads == 0 

        self.num_heads = num_heads

        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)

        self.proj = nn.Linear(dim,dim)


    def forward(self,x):
        B,N,C = x.shape

        qkv = self.qkv(x)

        qkv = qkv.reshape(
            B,
            N,
            3,
            self.num_heads,
            self.head_dim
        )

        qkv = qkv.permute(2, 0, 3, 1, 4)
        # (3,B,H,N,D)

        q,k,v = qkv[0], qkv[1], qkv[2]


        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)


        x = attn @ v

        x = x.transpose(1, 2)
        x = x.reshape(B, N, C)


        x = self.proj(x)

        return x 
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, n_classes=1000, 
                 embed_dim=768, depth=12, n_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + (img_size//patch_size)**2, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, n_heads) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
    def forward(self,x):
        x = self.patch_embed(x)

        B = x.shape[0]

        cls_token = self.cls_token.expand(B,-1,-1)

        x = torch.cat((cls_token,x), dim = 1)

        x = x + self.pos_embed


        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls = x[:, 0]
        # (B, 768)

        # 7. Classification Head
        x = self.head(cls)

        return x 
    