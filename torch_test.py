"""_summary_
Pytorch 에서 모델의 가중치를 저장하기 위해선 3가지 함수만 알면 충분 합니다.
    torch.save:
        객체를 디스크에 저장합니다. pickle 모듈을 이용하여 객체를 직렬화 하며, 
        이 함수를 사용하여 모든 종류의 모델, Tensor 등을 저장할 수 있습니다.
    torch.load: 
        pickle 모듈을 이용하여 객체를 역직렬화하여 메모리에 할당합니다.
    torch.nn.Module.load_state_dict:
        역직렬화된 state_dict를 사용, 모델의 매개변수들을 불러옵니다. 
        state_dict는 간단히 말해 각 체층을 매개변수 Tensor로 매핑한 Python 사전(dict) 객체입니다.
"""

import torch
import torch.nn as nn

x_data = torch.Tensor([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 0],
    [0, 0],
    [0, 1]
])

y_data = torch.LongTensor([
    0, # etc
    1, # mammal
    2, # bird
    0,
    0,
    2
])

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.w1 = nn.Linear(2, 10)
        self.bias1 = torch.zeros([10])
        
        self.w2 = nn.Linear(10, 3)
        self.bias2 = torch.zeros([3])
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, x):
        y = self.w1(x) + self.bias1
        y = self.relu(y)
        
        y = self.w2(y) + self.bias2
        return y

model = DNN()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    output = model(x_data)
    
    loss = criterion(output, y_data)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("progress:", epoch, "loss=", loss.item())


PATH = 'C:/home/freshfood/saved_models'
dict_path = PATH + '/model_state_dict.pt'
model_path = PATH + '/model.pt'
all_path = PATH + '/all.tar'

## 저장하기(CPU, GPU 동일)
torch.save(model, model_path) # 전체 모델 저장
torch.save(model.state_dict(), dict_path) # 모델객체의 state_dict 저장
torch.save({
    'model' : model.state_dict(),
    'optimizer' : optimizer.state_dict()
}, all_path) # 여러가지값 저장.학습중 진행상황저장을 위해 spoch,loss값등 일반scalar값 저장가능

## 불러오기
# 모델을 불러 온 이후에는 이 모델을 학습 할 껀지, 사용 할 껀지에 
# 따라 각각 model.train(), model.eval() 둘 중에 하나를 사용 하면 됩니다.
model = torch.load(model_path) # 전체모델을 통째로 불러옴, 클래스 선언 필수
model.load_state_dict(torch.load(dict_path)) # state_dict불러 온 후, 모델에 저장

checkpoint = torch.load(all_path) # dict 불러오기
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])


# ### 불러오기 GPU 저장 -> CPU 불러오기
# # torch.load() 함수의 map_location 인자에 torch.device('cpu') 를 전달 함으로써, 모델을 동적으로 CPU 장치에 할당합니다.
# device = torch.device('cpu')
# model = DNN()
# model.load_state_dict(torch.load(dict_path, map_location=device))

# ### 불러오기 CPU 저장 -> GPU 불러오기
# # torch.load() 로 초기화 한 모델의 model.to(torch.device('cuda')) 를 호출하여, CUDA Tensor로 내부 매개변수를 형변환 해 주어야 합니다.
# device = torch.device('cuda')
# model = DNN()
# model.load_state_dict(torch.load(dict_path))
# model.to(device)

# ### 불러오기 GPU 저장 -> GPU 불러오기
# # torch.load() 함수의 map_location 인자에 cuda:device_id 를 전달 함으로써, 
# # 모델을 동적으로 해당 GPU 장치에 할당합니다. 그 이후에 model.to(torch.device('cuda'))를 호출,
# # 모델 내의 Tensor를 CUDA Tensor로 변환 합니다. 모든 모델 입력에도 .to(torch.device('cuda'))를
# # 입력하여, CUDA Tensor로 변환하여야 합니다.
# device = torch.device('cuda')
# model = DNN()
# model.load_state_dict(torch.load(dict_path, map_location='cuda:0'))
# model.to(device)
