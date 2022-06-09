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
model = EfficientNet.from_pretrained(model_name, num_classes=10)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir    = f"{PROJECT_DIR}\dataset"
src_dir     = f"{data_dir}\img.jpg"
label_dir   = f'{data_dir}\labels_map.txt'

## 데이타 로드!!
batch_size  = 128
random_seed = 555
random.seed(random_seed)
torch.manual_seed(random_seed)

## make dataset
data_path = 'president/president_data'  # class 별 폴더로 나누어진걸 확 가져와서 라벨도 달아준다
president_dataset = datasets.ImageFolder(
                                data_path,
                                transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ]))
## data split
train_idx, tmp_idx = train_test_split(list(range(len(president_dataset))), test_size=0.2, random_state=random_seed)
datasets = {}
datasets['train'] = Subset(president_dataset, train_idx)
tmp_dataset       = Subset(president_dataset, tmp_idx)

val_idx, test_idx = train_test_split(list(range(len(tmp_dataset))), test_size=0.5, random_state=random_seed)
datasets['valid'] = Subset(tmp_dataset, val_idx)
datasets['test']  = Subset(tmp_dataset, test_idx)

## data loader 선언
dataloaders, batch_num = {}, {}
dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'],
                                              batch_size=batch_size, shuffle=True,
                                              num_workers=4)
dataloaders['valid'] = torch.utils.data.DataLoader(datasets['valid'],
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=4)
dataloaders['test']  = torch.utils.data.DataLoader(datasets['test'],
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=4)
batch_num['train'], batch_num['valid'], batch_num['test'] = len(dataloaders['train']), len(dataloaders['valid']), len(dataloaders['test'])
print('batch_size : %d,  tvt : %d / %d / %d' % (batch_size, batch_num['train'], batch_num['valid'], batch_num['test']))


img         = Image.open(src_dir)
# img.show()
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
