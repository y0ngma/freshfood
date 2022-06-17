import os, json, datetime
from PIL import Image
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# data_path   = f"{os.path.dirname(PROJECT_DIR)}/dataset/freshfood/all"
model_path  = f"{os.path.dirname(PROJECT_DIR)}/saved_models/freshfood"
data_dir    = f"{PROJECT_DIR}/img_cap" # C:/home/freshfood/img_cap
label_dir   = f'{data_dir}/labels_map.txt'
filename    = "pineapple.jpg"
img         = Image.open(f"{data_dir}/{filename}")
# img.show()
print('원본이미지크기', img.size)

# 캡쳐하기
if not os.path.isdir(data_dir): os.makedirs(data_dir)

# Preprocess image
tfms = transforms.Compose(
    [transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
     ])
img = tfms(img).unsqueeze(0)
print(img.shape)

model_name = 'efficientnet-b0'  # b5
image_size = EfficientNet.get_image_size(model_name)
# print(image_size)
model = EfficientNet.from_pretrained(model_name)
model = torch.load(f"{model_path}/20220617_epoch@4.pt")
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
