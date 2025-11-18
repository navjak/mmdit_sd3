# MMDiT-PyTorch
PyTorch implementation of the **Multimodal Diffusion Transformer (MMDiT)** from *Stable Diffusion 3*.

<!--
> This release currently includes only the original 2-modality MMDiT (text-image).  
> The extended **n-modality** version will be added in a future release.
-->

---

## Installation
<!--
```bash
pip install mmdit-pytorch
````

or from source:
-->

```bash
git clone https://github.com/navjak/mmdit-pytorch.git
cd mmdit-pytorch
pip install -e .
```

---

## Usage

```python
import torch
from mmdit_pytorch import MMDiTLayer

# define MMDiT layer
block = MMDiTLayer(
    dim_cond = 256,
    dim_text = 768,
    dim_image = 512,
    qk_rmsnorm = True
)

# mock inputs
time_cond = torch.randn(2, 256)

text_tokens = torch.randn(2, 512, 768)
text_mask = torch.ones((2, 512)).bool()

image_tokens = torch.randn(2, 1024, 512)

# single block forward
text_tokens_next, image_tokens_next = block(
    time_cond = time_cond,
    text_tokens = text_tokens,
    text_mask = text_mask,
    image_tokens = image_tokens
)
```

---

## Citations

```bibtex
@article{Esser2024ScalingRF,
    title   = {Scaling Rectified Flow Transformers for High-Resolution Image Synthesis},
    author  = {Patrick Esser and Sumith Kulal and A. Blattmann and Rahim Entezari and Jonas Muller and Harry Saini and Yam Levi and Dominik Lorenz and Axel Sauer and Frederic Boesel and Dustin Podell and Tim Dockhorn and Zion English and Kyle Lacey and Alex Goodwin and Yannik Marek and Robin Rombach},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2403.03206},
    url     = {https://api.semanticscholar.org/CorpusID:268247980}
}
```

```bibtex
@article{Zhu2024HyperConnections,
    title   = {Hyper-Connections},
    author  = {Defa Zhu and Hongzhi Huang and Zihao Huang and Yutao Zeng and Yunyao Mao and Banggu Wu and Qiyang Min and Xun Zhou},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2409.19606},
    url     = {https://api.semanticscholar.org/CorpusID:272987528}
}
```
