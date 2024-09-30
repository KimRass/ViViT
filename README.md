# ViViT: A Video Vision Transformer

# 1. How to Use
```python
from vivit import (
    SpatTempoAttnViViT, # Model 1: 'Spatio-temporal attention'
    FactorEncViViT, # Model 2: 'Factorised encoder'
    FactorSelfAttnViViT, # Model 3: 'Factorised self-attention'
)

# e.g.,
num_frames = 16
img_size = 224
video = torch.randn((4, 3, num_frames, img_size, img_size))

vit = timm.create_model("vit_base_patch16_224", pretrained=True)
tempo_patch_size = 4
spat_patch_size = 16
num_classes = 1000
pooling_mode="cls"
model = FactorEncViViT(
    vit=vit,
    num_frames=num_frames,
    img_size=img_size,
    tempo_patch_size=tempo_patch_size,
    spat_patch_size=spat_patch_size,
    num_classes=num_classes,
    pooling_mode=pooling_mode,
)

device = torch.device("cuda")
model = model.to(device)
video = video.to(device)

out = model(video) # (B, `num_classes`)
```

# 2. Citation
```bibtext
@misc{arnab2021vivit,
    title={ViViT: A Video Vision Transformer}, 
    author={Anurag Arnab and Mostafa Dehghani and Georg Heigold and Chen Sun and Mario Lučić and Cordelia Schmid},
    year={2021},
    eprint={2103.15691},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```