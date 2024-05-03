# References:
    # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    # https://github.com/rishikksh20/ViViT-pytorch/blob/master/vivit.py

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import timm


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
        self.conv3d.bias.data = vit.patch_embed.proj.bias.data
        self.to_seq = Rearrange("b d t h w -> b t (h w) d")

    def forward(self, x): # (B, C, T, H, W)
        x = self.conv3d(x) # (B, d, $n_{t}$, $n_{h}$, $n_{w}$)
        return self.to_seq(x)


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


class ViViTBase(nn.Module):
    def get_pos_embed_using_interpol(self):
        vit_img_size = self.vit.patch_embed.img_size
        vit_patch_size = self.vit.patch_embed.patch_size
        vit_pos_embed = self.vit.pos_embed
        vit_pos_embed_2d = vit_pos_embed[:, 1:, :].view(
            1,
            vit_img_size[0] // vit_patch_size[0],
            vit_img_size[1] // vit_patch_size[1],
            self.hidden_dim,
        )
        spat_pos_embed = F.interpolate(
            vit_pos_embed_2d.permute(0, 3, 1, 2),
            size=self.spat_seq_len,
        )
        return einops.rearrange(
            spat_pos_embed, pattern="1 d h w -> 1 (h w) d",
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
        num_frames,
        img_size,
        tempo_patch_size,
        spat_patch_size,
        num_classes,
        pooling_mode,
    ):
        super().__init__()

        self.vit = vit
        self.pooling_mode = pooling_mode
        self.img_size = img_size
        self.hidden_dim = vit.embed_dim
        self.spat_patch_size = spat_patch_size

        self.tublet_embed = TubletEmbedding(
            vit=vit,
            tempo_patch_size=tempo_patch_size,
            spat_patch_size=spat_patch_size,
            hidden_dim=self.hidden_dim,
        )
        self.merge_spat_tempo = Rearrange("b t s d -> b (t s) d")

        self.spat_seq_len = img_size // spat_patch_size
        self.tempo_seq_len = num_frames // tempo_patch_size
        self.pos_embed = self.get_pos_embed_using_interpol()
        self.vit_blocks = vit.blocks
        self.mlp_head = nn.Linear(self.hidden_dim, num_classes)

    @staticmethod
    def batch_to_tempo(x, batch_size):
        if x.ndim == 2:
            pattern = "(b t) d -> b t d"
        elif x.ndim == 3:
            pattern = "(b t) s d -> b t s d"
        return einops.rearrange(x, pattern=pattern, b=batch_size)

    @staticmethod
    def batch_to_spat(x, batch_size):
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


class SpatTempoAttnViViT(ViViTBase):
    """
    "Each transformer layer models all pairwise interactions between all
    spatio-temporal tokens, and it thus models long-range interactions
    across the video from the first layer."
    "Our video models have nt times more tokens than the pretrained image model.
    As a result, we initialise the positional embeddings by “repeating” them
    temporally from Rnw ·nh ×d to Rnt·nh·nw ×d . Therefore, at initialisation, all tokens with the same spatial index have the same embedding which is then fine-tuned.
    """
    def __init__(
        self,
        vit,
        num_frames,
        img_size,
        tempo_patch_size,
        spat_patch_size,
        num_classes,
        pooling_mode="cls",
    ):
        super().__init__(
            vit=vit,
            num_frames=num_frames,
            img_size=img_size,
            tempo_patch_size=tempo_patch_size,
            spat_patch_size=spat_patch_size,
            pooling_mode=pooling_mode,
            num_classes=num_classes,
        )

        # self.cls_token = vit.cls_token
        # self.cls_pos_embed = vit.pos_embed[:, 0: 1, :]
        self.cls_token = nn.Parameter(torch.randn((1, 1, self.hidden_dim)))
        self.cls_pos_embed = nn.Parameter(torch.randn((1, 1, self.hidden_dim)))
        self.pos_embed = einops.repeat(
            self.pos_embed, pattern="1 s d -> 1 t s d", t=self.tempo_seq_len,
        )
        self.prepend_cls_token = lambda x: torch.cat(
            (self.cls_token.repeat(x.size(0), 1, 1), x), dim=1,
        )
        self.transformer = self.vit_blocks

    def forward(self, x):
        x = self.tublet_embed(x)
        x = self.merge_spat_tempo(x)
        x = self.prepend_cls_token(x)
        x += torch.cat(
            [self.cls_pos_embed, self.merge_spat_tempo(self.pos_embed)], dim=1,
        )
        x = self.transformer(x)
        x = self.pool(x)
        return self.mlp_head(x)


class FactorEncViViT(ViViTBase):
    """
    "The first, spatial encoder, only models interactions between tokens
    extracted from the same temporal index."
    "A representa- tion for each temporal index, hi ∈ Rd, is obtained after Ls layers: This is the encoded classification token, zcls Ls if it was prepended to the input (Eq. 1), or a global average pooling from the tokens output by the spatial encoder, zLs , otherwise."
    "The frame-level representations, $h_{i}$, are concatenated into
    $H \in \mathbb{R}^{n_{t} \times d}$, and then forwarded through a temporal encoder consisting of Lt transformer layers to model in- teractions between tokens from different temporal indices.
    "The initial spatial encoder is identical to the one used for image classification."
    """
    def __init__(
        self,
        vit,
        num_frames,
        img_size,
        tempo_patch_size,
        spat_patch_size,
        num_classes,
        pooling_mode="cls",
    ):
        super().__init__(
            vit=vit,
            num_frames=num_frames,
            img_size=img_size,
            tempo_patch_size=tempo_patch_size,
            spat_patch_size=spat_patch_size,
            pooling_mode=pooling_mode,
            num_classes=num_classes,
        )

        self.spat_cls_token = nn.Parameter(
            torch.randn((1, 1, self.hidden_dim)),
        )
        self.prepend_spat_cls_token = lambda x: torch.cat(
            (self.spat_cls_token.repeat(x.size(0), 1, 1), x), dim=1,
        )
        self.spat_pos_embed = self.pos_embed
        self.spat_cls_pos_embed = nn.Parameter(
            torch.randn((1, 1, self.hidden_dim)),
        )
        self.tempo_to_batch = Rearrange("b t s d -> (b t) s d")

        self.tempo_cls_token = nn.Parameter(torch.randn((1, 1, self.hidden_dim)))
        self.prepend_tempo_cls_token = lambda x: torch.cat(
            (self.tempo_cls_token.repeat(x.size(0), 1, 1), x), dim=1,
        )
        self.tempo_pos_embed = nn.Parameter(
            torch.randn((1, self.tempo_seq_len, self.hidden_dim))
        )
        self.tempo_cls_pos_embed = nn.Parameter(
            torch.randn((1, 1, self.hidden_dim)),
        )

        self.spat_encoder = self.vit_blocks
        self.tempo_encoder = nn.Sequential(
            *([self.spat_encoder[0]] * len(self.spat_encoder)),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.tublet_embed(x)

        x = self.tempo_to_batch(x) # "$h_{i}$"
        x = self.prepend_spat_cls_token(x)
        x += torch.cat(
            [self.spat_cls_pos_embed, self.spat_pos_embed], dim=1,
        )
        x = self.spat_encoder(x)
        x = self.pool(x)

        x = self.batch_to_tempo(x, batch_size=batch_size)
        x = self.prepend_tempo_cls_token(x)
        x += torch.cat(
            [self.tempo_cls_pos_embed, self.tempo_pos_embed], dim=1,
        )
        x = self.tempo_encoder(x) # "$H$"
        x = self.pool(x)
        return self.mlp_head(x)


class FactorSelfAttnViViT(ViViTBase):
    """
    "We factorise the operation to first only compute self-attention spatially (among all tokens extracted from the same tem- poral index), and then temporally (among all tokens ex- tracted from the same spatial index)"
    "We do not use a classification token in this model, to avoid
    ambiguities when reshaping the input tokens between spatial and
    temporal dimensions.
    """
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
        num_frames,
        img_size,
        tempo_patch_size,
        spat_patch_size,
        num_classes,
        pooling_mode="cls",
    ):
        super().__init__(
            vit=vit,
            num_frames=num_frames,
            img_size=img_size,
            tempo_patch_size=tempo_patch_size,
            spat_patch_size=spat_patch_size,
            pooling_mode=pooling_mode,
            num_classes=num_classes,
        )

        self.tempo_to_batch = Rearrange("b t s d -> (b t) s d")
        self.spat_to_batch = Rearrange("b t s d -> (b s) t d")
        self.spat_blocks = self.vit_blocks
        self.tempo_blocks = nn.Sequential(
            *(
                [
                    MSA(
                        hidden_dim=self.hidden_dim,
                        num_heads=num_heads,
                        drop_prob=drop_prob,
                    ),
                    nn.LayerNorm(self.hidden_dim),
                ] * len(self.spat_blocks)
            )
        )
        self.init_tempo_blocks()
        self.pooling_mode == "avg"

    def forward(self, x):
        batch_size = x.size(0)
        x = self.tublet_embed(x)

        for spat_block, tempo_block in zip(self.spat_blocks, self.tempo_blocks):
            x = self.tempo_to_batch(x)
            x = x + spat_block.drop_path1(
                spat_block.ls1(spat_block.attn(spat_block.norm1(x)))
            )
            x = self.untempo_to_batch(x, batch_size=batch_size)

            x = self.spat_to_batch(x)
            # temp = x
            x = x + tempo_block(x)[0]
            # print(torch.equal(x, temp))
            x = self.unspat_to_batch(x, batch_size=batch_size)

            x = x + spat_block.drop_path2(
                spat_block.ls2(spat_block.mlp(spat_block.norm2(x)))
            )
        x = self.pool(x)
        return self.mlp_head(x)


if __name__ == "__main__":
    num_frames = 8
    img_size = 64
    tempo_patch_size=2
    spat_patch_size=16

    device = torch.device("cuda")
    vit = timm.create_model("vit_base_patch16_224", pretrained=True).to(device)
    # model = SpatTempoAttnViViT(
    model = FactorEncViViT(
        vit=vit,
        num_frames=num_frames,
        img_size=img_size,
        tempo_patch_size=tempo_patch_size,
        spat_patch_size=spat_patch_size,
        pooling_mode="cls",
    ).to(device)
    
    video = torch.randn((4, 3, num_frames, img_size, img_size)).to(device)
    x = model(video)
    x.shape
