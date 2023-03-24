# clip-konw
![clip](https://github.com/openai/CLIP/blob/main/CLIP.png?raw=true "clip")

我的理解：CLIP（对比性语言-图像预训练）是一个在各种（图像，文本）对上训练的神经网络。它可以在自然语言中被指示预测最相关的文本片段，给定一个图像，而不直接对任务进行优化， CLIP 真的把自然语言级别的抽象概念带到计算机视觉里了。
CLIP是将Text Decoder从文本中提取的语义特征和Image Decoder从图像中提取的语义特征进行匹配训练

## 安装
```phthon
pip install ftfty regex tqdmhttps://github.com/openai/CLIP/blob/main/CLIP.png?raw=true
pip install git+https://github.com/openai/CLIP.gitv
```
## Usage
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
## API
```python
import os
import clip
import torch
from PIL import Image
from torchvision.datasets import CIFAR100

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
#cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
image = preprocess(Image.open("红包.png")).unsqueeze(0).to(device)
# Prepare the inputs


list=['red', 'envelope', 'China']
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in list]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(3)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{list[index]:>16s}: {100 * value.item():.2f}%")
```
```python
Top predictions:

           China: 72.51%
        envelope: 14.06%
             red: 13.42%
```
