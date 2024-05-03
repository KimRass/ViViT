

class ViViT(nn.Module):
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
        pos_embed_2d = F.interpolate(
            vit_pos_embed_2d.permute(0, 3, 1, 2),
            size=img_size // self.spat_patch_size,
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
        num_heads=12,
        drop_prob=0.1,
    ):
        super().__init__()

        self.vit = vit
        self.mode = mode
        self.pooling_mode = pooling_mode
        self.hidden_dim = vit.embed_dim
        self.spat_patch_size = spat_patch_size

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
        self.pos_embed = self.get_pos_embed_using_interpol()
        # x = torch.cat(
        #     [einops.repeat(cls_token, pattern="d -> b 1 1 d", b=x.size(0)), x],
        #     dim=1,
        # )
        vit_blocks = vit.blocks
        self.mlp_head = nn.Linear(self.hidden_dim, num_classes)

        if self.mode == "spat_tempo_attn":
            self.merge_spat_tempo = Rearrange("b t s d -> b (t s) d")
            self.transformer = vit_blocks
            self.unmerge_spat_tempo = Rearrange("b (t s) d -> b t s d")

        if self.mode == "factor_encoder":
            self.tempo_to_batch = Rearrange("b t s d -> (b t) s d")
            self.num_spat_layers = len(vit_blocks)
            self.spat_encoder = vit_blocks
            self.tempo_encoder = nn.Sequential(
                *([vit_blocks[0]] * num_tempo_layers),
            )

        if self.mode == "factor_self_attn":
            
            self.tempo_to_batch = Rearrange("b t s d -> (b t) s d")
            self.spat_to_batch = Rearrange("b t s d -> (b s) t d")
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
    def untempo_to_batch(x, batch_size):
        if x.ndim == 2:
            pattern = "(b t) d -> b t d"
        elif x.ndim == 3:
            pattern = "(b t) s d -> b t s d"
        return einops.rearrange(x, pattern=pattern, b=batch_size)

    @staticmethod
    def unspat_to_batch(x, batch_size):
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
            x = self.merge_spat_tempo(x)
            x = self.prepend_cls_token(x)
            x += self.pos_embed
            x = self.transformer(x)
            x = self.unmerge_spat_tempo(x)
            x = self.pool(x)

        elif self.mode == "factor_encoder":
            x = self.tempo_to_batch(x) # "$h_{i}$"
            x = self.spat_encoder(x)
            x = self.pool(x)
            x = self.untempo_to_batch(x, batch_size=batch_size)
            x = self.tempo_encoder(x) # "$H$"
            x = self.pool(x)

        elif self.mode == "factor_self_attn":
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

        # x = self.mlp_head(x)
        return x