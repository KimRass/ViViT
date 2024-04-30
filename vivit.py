# References:
    # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops.layers.torch import Rearrange
from typing import Optional
import timm
from timm.models.vision_transformer import (
    Attention,
    Mlp,
    LayerScale,
    DropPath,
)

# from vit import ViT


"""
"Our video models have nt times more tokens than the pretrained image model.
As a result, we initialise the positional embeddings by “repeating” them
temporally from Rnw ·nh ×d to Rnt·nh·nw ×d . Therefore, at initialisation, all tokens with the same spatial index have the same embedding which is then fine-tuned.
"""
spat_patch_size = 16
tempo_patch_size = 2
vit_pos_embed = vit.pos_embed
pos_embed = einops.repeat(
    vit_pos_embed, pattern="1 l d -> 1 (l n) d", n=tempo_patch_size,
)


class TubletEmbedding(nn.Module):
    """
    "Extract non-overlapping, spatio-temporal “tubes” from the input volume, and to linearly project this to Rd. For a tubelet of di- mension t × h × w, nt = b Tt c, nh = b H h c and nw = b W w c, tokens are extracted from the temporal, height, and width dimensions respectively. Smaller tubelet dimensions thus result in more tokens which increases the computation. This method fuses spatio-temporal information during tokenisation, in contrast to “Uniform frame sam- pling” where temporal information from different frames is fused by the transformer.
    "We denote as 'central frame initialisation', where E is initialised with ze-
    roes along all temporal positions, except at the centre b t c,
    2E = [0, . . . , Eimage, . . . , 0]
    """
    def __init__(self, vit, tempo_patch_size, spat_patch_size, hidden_dim):
        super().__init__()

        kernel_size = stride = (tempo_patch_size, spat_patch_size, spat_patch_size)
        self.conv3d = nn.Conv3d(3, hidden_dim, kernel_size, stride, 0)
        vit_patch_embed = vit.patch_embed
        self.conv3d.bias.data = vit_patch_embed.proj.bias.data
        self.to_seq = Rearrange("b l t h w -> b l t (h w)")

    def forward(self, x):
        x = self.conv3d(x)
        x = self.to_seq(x)
        return x.permute(0, 2, 3, 1)


# class Block(nn.Module):
#     def __init__(
#             self,
#             dim: int,
#             num_heads: int,
#             mlp_ratio: float = 4.,
#             qkv_bias: bool = False,
#             qk_norm: bool = False,
#             proj_drop: float = 0.,
#             attn_drop: float = 0.,
#             init_values: Optional[float] = None,
#             drop_path: float = 0.,
#             act_layer: nn.Module = nn.GELU,
#             norm_layer: nn.Module = nn.LayerNorm,
#             mlp_layer: nn.Module = Mlp,
#     ) -> None:
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(
#             dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_norm=qk_norm,
#             attn_drop=attn_drop,
#             proj_drop=proj_drop,
#             norm_layer=norm_layer,
#         )
#         self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
#         self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#         self.norm2 = norm_layer(dim)
#         self.mlp = mlp_layer(
#             in_features=dim,
#             hidden_features=int(dim * mlp_ratio),
#             act_layer=act_layer,
#             drop=proj_drop,
#         )
#         self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
#         self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
#         # x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
#         x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
#         return x


class MSA(nn.Module):
    def __init__(self, hidden_dim, num_heads, drop_prob):
        super().__init__()
    
        self.num_heads = num_heads

        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.to_multi_heads = Rearrange("b i (n h) -> b i n h", n=num_heads)
        self.scale = hidden_dim ** (-0.5)
        self.attn_drop = nn.Dropout(drop_prob)
        self.to_one_head = Rearrange("b i n h -> b i (n h)")
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, mask=None):
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
        attn_score = torch.einsum(
            "binh,bjnh->bnij", self.to_multi_heads(q), self.to_multi_heads(k),
        ) * self.scale
        if mask is not None:
            attn_score.masked_fill_(
                mask=einops.repeat(
                    mask, pattern="b i j -> b n i j", n=self.num_heads,
                ),
                value=-1e9,
            )
        attn_weight = F.softmax(attn_score, dim=-1)
        x = self.to_one_head(
            torch.einsum(
                "bnij,bjnh->binh",
                self.attn_drop(attn_weight),
                self.to_multi_heads(v),
            )
        )
        x = self.out_proj(x)
        return x, attn_weight


class ViViT(nn.Module):
    def __init__(
        self,
        vit,
        mode,
        img_size=224,
        tempo_patch_size=2,
        spat_patch_size=16,
        num_tempo_layers=12,
        num_spat_layers=12,
        hidden_dim=768,
        mlp_size=3072,
        num_heads=12,
        drop_prob=0.1,
        n_classes=0,
    ):
        super().__init__()

        self.mode = mode

        self.tuplet_embed = TubletEmbedding(
            vit=vit,
            tempo_patch_size=tempo_patch_size,
            spat_patch_size=spat_patch_size,
            hidden_dim=hidden_dim,
        )
        # cls_token = nn.Parameter(torch.randn((hidden_dim)))
        # x = torch.cat(
        #     [einops.repeat(cls_token, pattern="d -> b 1 1 d", b=x.size(0)), x],
        #     dim=1,
        # )

        self.spat_blocks = vit.blocks
        self.hide_temporal = Rearrange("b t s d -> (b t) s d")
        self.hide_spatial = Rearrange("b t s d -> (b s) t d")
        self.to_4d = lambda x: x.view(self.ori_shape)

        if self.mode == "factor_self_attn":
            """
            "It contains two multi-headed self atten- tion (MSA) modules. We initialise the spatial MSA module from the pretrained module, and initialise all weights of the temporal MSA with zeroes, such that Eq. 5 behaves as a residual connection at initialisation.
            """
            self.tempo_blocks = nn.ModuleList(
                [
                    MSA(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        drop_prob=drop_prob,
                    ),
                    nn.LayerNorm(hidden_dim),
                ] * len(self.spat_blocks)
            )

    def forward(self, x):
        x = self.tuplet_embed(x)

        self.ori_shape = x.shape
        x = x.contiguous()

        if self.mode == "factor_encoder":
            """
            e. The frame-level representations, hi , are concatenated into H ∈ Rnt×d , and then forwarded through a temporal encoder consisting of Lt transformer layers to model in- teractions between tokens from different temporal indices.
            "The initial spatial encoder is identical to the one used for image classification."
            """
            x = self.hide_temporal(x)
            x = self.spat_blocks(x)
            x = self.to_4d(x)

        elif self.mode == "factor_self_attn":
            """
            "We factorise the operation to first only compute self-attention spatially (among all tokens extracted from the same tem- poral index), and then temporally (among all tokens ex- tracted from the same spatial index)"
            """
            x = self.hide_temporal(x)
            for spat_block, tempo_block in zip(self.spat_blocks, self.tempo_blocks):
                x = x + spat_block.drop_path1(
                    spat_block.ls1(spat_block.attn(spat_block.norm1(x)))
                )
                x = x + tempo_block(x)[0]
                x = x + spat_block.drop_path2(
                    spat_block.ls2(spat_block.mlp(spat_block.norm2(x)))
                )
            x = self.to_4d(x)
        return x
img_size = 64
vid_len = 16
hidden_dim = 768

device = torch.device("cuda")
vit = timm.create_model("vit_base_patch16_224", pretrained=True).to(device)
# model = ViViT(vit=vit, mode="factor_encoder").to(device)
model = ViViT(vit=vit, mode="factor_self_attn").to(device)
video = torch.randn((4, 3, vid_len, img_size, img_size)).to(device)
x = model(video)
x.shape
