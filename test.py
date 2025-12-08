from diffusers import PixArtAlphaPipeline
import torch

pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-512x512", 
    torch_dtype=torch.float16
)
pipe.to("cuda")

image = pipe("A cat wearing a hat").images[0]

import matplotlib.pyplot as plt
plt.imshow()
plt.savefig("cat_wearing_hat.png")