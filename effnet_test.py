import os, json, datetime
from PIL import Image
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import cv2
import my_utils


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# data_path   = f"{os.path.dirname(PROJECT_DIR)}/dataset/freshfood/all"
model_path  = f"{os.path.dirname(PROJECT_DIR)}/saved_models/freshfood"
data_dir    = f"{PROJECT_DIR}/img_cap" # C:/home/freshfood/img_cap
label_dir   = f'{data_dir}/labels_map.txt'

# Load ImageNet class names
labels_map = json.load(open(label_dir))
labels_map = [labels_map[str(i)] for i in range(len(labels_map))]

## 모델 로드
model_name = 'efficientnet-b0'  # b5
model      = EfficientNet.from_pretrained(model_name, num_classes=10)
model.load_state_dict(torch.load(f"{model_path}/20220617_epoch@4.pt"))
model.eval()


## 캡쳐하기
if not os.path.isdir(data_dir): os.makedirs(data_dir)
# filename    = "pineapple.jpg"
# img         = Image.open(f"{data_dir}/{filename}")
img_dir = my_utils.capture_vid(data_dir)

## Preprocess image
img         = Image.open(img_dir)
# img.show()
print('원본 이미지크기', img.size)
tfms = transforms.Compose(
    [transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
     ])
img = tfms(img).unsqueeze(0)
image_size = EfficientNet.get_image_size(model_name)
print("EfficientNet.get_image_size(model_name) = ", image_size)
features   = model.extract_features(img)
print("model.extract_features(img).shape = ", features.shape)
print("img.shape = ", img.shape)


# Classify
start_time = datetime.datetime.now() # 시간측정
with torch.no_grad():
    outputs = model(img)

# Print predictions
print('---------')
for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
    prob = torch.softmax(outputs, dim=1)[0, idx].item()
    print('{label:<75} ({p:0>6.2f}%)'.format(label=labels_map[idx], p=prob*100))
print("소요시간=> ", datetime.datetime.now() - start_time)
