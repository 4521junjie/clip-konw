# clip-konw
[baidu](http://www.baidu.com/img/bdlogo.gif "百度logo")

## 安装
```phthon
pip install ftfty regex tqdmhttps://github.com/openai/CLIP/blob/main/CLIP.png?raw=true
pip install git+https://github.com/openai/CLIP.gitv
```
## APL
```python
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("红包.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a red envelope", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
```

