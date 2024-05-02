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

class TubletEmbedding(nn.Module):
    """
    "We consider two simple methods for mapping a video V ∈ RT ×H×W ×C to a sequence of tokens z̃ ∈ Rnt ×nh ×nw ×d. We then add the positional embedding and N ×dreshape into R to obtain z, the input to the transformer."
    
    "Extract non-overlapping, spatio-temporal “tubes” from the input volume, and to linearly project this to Rd. For a tubelet of di- mension t × h × w, nt = b Tt c, nh = b H h c and nw = b W w c, tokens are extracted from the temporal, height, and width dimensions respectively. Smaller tubelet dimensions thus result in more tokens which increases the computation. This method fuses spatio-temporal information during tokenisation, in contrast to “Uniform frame sam- pling” where temporal information from different frames is fused by the transformer.
    "We denote as 'central frame initialisation', where E is initialised with ze-
    roes along all temporal positions, except at the centre b t c,
    2E = [0, . . . , Eimage, . . . , 0]
    """
    def __init__(self, vit, tempo_patch_size, spat_patch_size, hidden_dim):
        super().__init__()

        kernel_size = stride = (
            tempo_patch_size, spat_patch_size, spat_patch_size,
        ) # "$t \times h \times h$"
        self.conv3d = nn.Conv3d(3, hidden_dim, kernel_size, stride, 0)
        vit_patch_embed = vit.patch_embed
        self.conv3d.bias.data = vit_patch_embed.proj.bias.data
        self.to_seq = Rearrange("b l t h w -> b l t (h w)")

    def forward(self, x): # (B, C, T, H, W)
        x = self.conv3d(x) # (B, d, $n_{t}$, $n_{h}$, $n_{w}$)
        x = self.to_seq(x)
        return x.permute(0, 2, 3, 1)


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
    def get_pos_embed_using_interpol(self, vit):
        vit_img_size = vit.patch_embed.img_size
        vit_patch_size = vit.patch_embed.patch_size
        vit_pos_embed_2d = vit_pos_embed[:, 1:, :].view(
            1,
            vit_img_size[0] // vit_patch_size[0],
            vit_img_size[1] // vit_patch_size[1],
            self.hidden_dim,
        )
        pos_embed_2d = F.interpolate(
            vit_pos_embed_2d.permute(0, 3, 1, 2),
            size=img_size // spat_patch_size,
        )
        pos_embed_1d = einops.rearrange(
            pos_embed_2d, pattern="1 d h w -> 1 (h w) d",
        )
        return torch.cat(
            [vit_pos_embed[:, 0: 1, :], pos_embed_1d], dim=1,
        )

    def init_tempo_blocks(self):
        """
        "It contains two multi-headed self attention (MSA) modules. We
        initialise the spatial MSA module from the pretrained module, and
        initialise all weights of the temporal MSA with zeroes, such that Eq. 5
        behaves as a residual connection at initialisation."
        """
        for layer in self.tempo_blocks:
            for param in layer.parameters():
                nn.init.constant_(param, 0)

    def __init__(
        self,
        vit,
        mode,
        pooling_mode="cls",
        num_classes=100,
        img_size=224,
        tempo_patch_size=2,
        spat_patch_size=16,
        num_tempo_layers=12,
        num_spat_layers=12,
        mlp_size=3072,
        num_heads=12,
        drop_prob=0.1,
    ):
        super().__init__()

        self.mode = mode
        self.pooling_mode = pooling_mode
        self.hidden_dim = vit.embed_dim

        self.tuplet_embed = TubletEmbedding(
            vit=vit,
            tempo_patch_size=tempo_patch_size,
            spat_patch_size=spat_patch_size,
            hidden_dim=self.hidden_dim,
        )
        self.cls_token = vit.cls_token
        self.prepend_cls_token = lambda x: torch.cat(
            (self.cls_token.repeat(x.size(0), 1, 1), x), dim=1,
        )
        self.pos_embed = self.get_pos_embed_using_interpol(vit=vit)
        # x = torch.cat(
        #     [einops.repeat(cls_token, pattern="d -> b 1 1 d", b=x.size(0)), x],
        #     dim=1,
        # )
        self.mlp_head = nn.Linear(self.hidden_dim, num_classes)

        if self.mode == "spat_tempo_attn":
            """
            "Each transformer layer models all pairwise interactions between all
            spatio-temporal tokens, and it thus models long-range interactions
            across the video from the first layer."
            """
            self.merge = Rearrange("b t s d -> b (t s) d")
            self.unmerge = Rearrange("b (t s) d -> b t s d")

        if self.mode == "factor_encoder":
            """
            "The first, spatial encoder, only models interactions between tokens
            extracted from the same temporal index."
            "A representa- tion for each temporal index, hi ∈ Rd, is obtained after Ls layers: This is the encoded classification token, zcls Ls if it was prepended to the input (Eq. 1), or a global average pooling from the tokens output by the spatial encoder, zLs , otherwise."
            "The frame-level representations, $h_{i}$, are concatenated into
            $H \in \mathbb{R}^{n_{t} \times d}$, and then forwarded through a temporal encoder consisting of Lt transformer layers to model in- teractions between tokens from different temporal indices.
            "The initial spatial encoder is identical to the one used for image classification."
            """
            self.fold_tempo = Rearrange("b t s d -> (b t) s d")
            vit_blocks = vit.blocks
            self.num_spat_layers = len(vit_blocks)
            self.spat_encoder = vit_blocks
            self.tempo_encoder = nn.Sequential(
                *([vit_blocks[0]] * num_tempo_layers),
            )

        if self.mode == "factor_self_attn":
            """
            "We factorise the operation to first only compute self-attention spatially (among all tokens extracted from the same tem- poral index), and then temporally (among all tokens ex- tracted from the same spatial index)"
            "We do not use a classification token in this model, to avoid
            ambiguities when reshaping the input tokens between spatial and
            temporal dimensions.
            """
            self.fold_tempo = Rearrange("b t s d -> (b t) s d")
            self.fold_spat = Rearrange("b t s d -> (b s) t d")
            vit_blocks = vit.blocks
            self.spat_blocks = vit_blocks
            self.tempo_blocks = nn.Sequential(
                *(
                    [
                        MSA(
                            hidden_dim=self.hidden_dim,
                            num_heads=num_heads,
                            drop_prob=drop_prob,
                        ),
                        nn.LayerNorm(self.hidden_dim),
                    ] * len(vit_blocks)
                )
            )
            self.init_tempo_blocks()
            self.pooling_mode == "avg"

    @staticmethod
    def unfold_tempo(x, batch_size):
        if x.ndim == 2:
            pattern = "(b t) d -> b t d"
        elif x.ndim == 3:
            pattern = "(b t) s d -> b t s d"
        return einops.rearrange(x, pattern=pattern, b=batch_size)

    @staticmethod
    def unfold_spat(x, batch_size):
        # if x.ndim == 2:
        #     pattern = "(b t) d -> b t d"
        # elif x.ndim == 3:
        pattern = "(b s) t d -> b s t d"
        return einops.rearrange(x, pattern=pattern, b=batch_size)

    def pool(self, x):
        if self.pooling_mode == "cls":
            if x.ndim == 3:
                return x[:, 0, :]
            elif x.ndim == 4:
                return x[:, 0, 0, :]
        elif self.pooling_mode == "avg":
            if x.ndim == 3:
                return torch.mean(x, dim=1)
            elif x.ndim == 4:
                return torch.mean(x, dim=(1, 2))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.tuplet_embed(x)

        x = x.contiguous() # (batch_size, tempo, spat, hidden_dim)

        if self.mode == "spat_tempo_attn":
            x = self.merge(x)
            x = self.prepend_cls_token(x)
            x += self.pos_embed
            x = vit.blocks(x)
            x = self.unmerge(x)
            x = self.pool(x)

        elif self.mode == "factor_encoder":
            x = self.fold_tempo(x) # "$h_{i}$"
            x = self.spat_encoder(x)
            x = self.pool(x)
            x = self.unfold_tempo(x, batch_size=batch_size)
            x = self.tempo_encoder(x) # "$H$"
            x = self.pool(x)

        elif self.mode == "factor_self_attn":
            for spat_block, tempo_block in zip(self.spat_blocks, self.tempo_blocks):
                x = self.fold_tempo(x)
                x = x + spat_block.drop_path1(
                    spat_block.ls1(spat_block.attn(spat_block.norm1(x)))
                )
                x = self.unfold_tempo(x, batch_size=batch_size)

                x = self.fold_spat(x)
                # temp = x
                x = x + tempo_block(x)[0]
                # print(torch.equal(x, temp))
                x = self.unfold_spat(x, batch_size=batch_size)

                x = x + spat_block.drop_path2(
                    spat_block.ls2(spat_block.mlp(spat_block.norm2(x)))
                )
            x = self.pool(x)

        # x = self.mlp_head(x)
        return x


if __name__ == "__main__":
    img_size = 64
    vid_len = 16

    device = torch.device("cuda")
    vit = timm.create_model("vit_base_patch16_224", pretrained=True).to(device)
    # model = ViViT(vit=vit, pooling_mode="cls", mode="spat_tempo_attn").to(device)
    # model = ViViT(vit=vit, mode="factor_encoder").to(device)
    model = ViViT(vit=vit, mode="factor_self_attn").to(device)
    video = torch.randn((4, 3, vid_len, img_size, img_size)).to(device)
    x = model(video)
    x.shape
