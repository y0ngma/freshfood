import os, json, datetime, time, copy, random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Subset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from efficientnet_pytorch import EfficientNet

model_name = 'efficientnet-b0'  # b5
image_size = EfficientNet.get_image_size(model_name)
print(image_size)
model = EfficientNet.from_pretrained(model_name)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir    = f"{PROJECT_DIR}\dataset"
src_dir     = f"{data_dir}\img.jpg"
label_dir   = f'{data_dir}\labels_map.txt'
img         = Image.open(src_dir)
img.show()
print('원본이미지크기', img.size)

# Preprocess image
tfms = transforms.Compose(
    [transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
     ])
img = tfms(img).unsqueeze(0)
print(img.shape)

features = model.extract_features(img)
print(features.shape)

# Load ImageNet class names
labels_map = json.load(open(label_dir))
labels_map = [labels_map[str(i)] for i in range(1000)]

# 시간측정
start_time = datetime.datetime.now()

# Classify
model.eval()
with torch.no_grad():
    outputs = model(img)

print("소요시간=> ", datetime.datetime.now() - start_time)
    
# Print predictions
print('---------')
for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
    prob = torch.softmax(outputs, dim=1)[0, idx].item()
    print('{label:<75} ({p:0>6.2f}%)'.format(label=labels_map[idx], p=prob*100))
