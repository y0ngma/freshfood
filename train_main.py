## 학습 코드
import numpy as np
import json
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import random
from efficientnet_pytorch import EfficientNet

class_names = {
    "0": "apple",
    "1": "banana",
    "2": "carrot",
    "3": "cauliflower",
    "4": "garlic",
    "5": "ginger",
    "6": "grapes",
    "7": "mango",
    "8": "pineapple",
    "9": "watermelon"
}
model_name = 'efficientnet-b0'  # b5
model = EfficientNet.from_pretrained(model_name, num_classes=len(class_names))
print(EfficientNet.get_image_size(model_name))

# print("fc 제외하고 freeze")
# for n, p in model.named_parameters():
#     if '_fc' not in n:
#         p.requires_grad = False
# model = torch.nn.parallel.DistributedDataParallel(model)

batch_size  = 32
random_seed = 555
random.seed(random_seed)
torch.manual_seed(random_seed)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
data_path   = f"{os.path.dirname(PROJECT_DIR)}/dataset/{os.path.basename(PROJECT_DIR)}"

## make dataset
from torchvision import transforms, datasets
president_dataset = datasets.ImageFolder(
                                data_path,
                                transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ]))
## data split
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
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

## 데이타 체크
import torchvision
def imshowt(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

num_show_img = 5
class_names  = {"0": "apple","1": "banana","2": "carrot","3": "cauliflower","4": "garlic","5": "ginger","6": "grapes","7": "mango","8": "pineapple","9": "watermelon"}

print(dataloaders['train'])
print(iter(dataloaders['train']))

# The "freeze_support()" line can be omitted if the program is not going to be frozen to produce an executable.


# # train check
# inputs, classes = next(iter(dataloaders['train']))
# print(inputs, classes)
# out = torchvision.utils.make_grid(inputs[:num_show_img])  # batch의 이미지를 오려부친다
# imshowt(out, title=[class_names[str(int(x))] for x in classes[:num_show_img]])

# # valid check
# inputs, classes = next(iter(dataloaders['valid']))
# out = torchvision.utils.make_grid(inputs[:num_show_img])  # batch의 이미지를 오려부친다
# imshowt(out, title=[class_names[str(int(x))] for x in classes[:num_show_img]])

# # test check
# inputs, classes = next(iter(dataloaders['test']))
# out = torchvision.utils.make_grid(inputs[:num_show_img])  # batch의 이미지를 오려부친다
# imshowt(out, title=[class_names[str(int(x))] for x in classes[:num_show_img]])
