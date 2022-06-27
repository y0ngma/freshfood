## 학습 코드
from cProfile import label
import json, os, copy, random, time, datetime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from multiprocessing import freeze_support

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Subset
import cv2

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


def train_model(save_path, model, criterion, optimizer, scheduler, num_epochs=4):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss, running_corrects, num_cnt = 0.0, 0, 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_cnt += len(labels)
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = float(running_loss / num_cnt)
            epoch_acc  = float((running_corrects.double() / num_cnt).cpu()*100)
            
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
            print('{} Loss: {:.2f} Acc: {:.1f}'.format(phase, epoch_loss, epoch_acc))
           
            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_idx = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
#                 best_model_wts = copy.deepcopy(model.module.state_dict())
                print('==> best model saved - %d / %.1f'%(best_idx, best_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: %d - %.1f' %(best_idx, best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    model_save_name = f'{datetime.datetime.today().strftime("%Y%m%d")}_epoch@{best_idx}.pt'
    torch.save(model.state_dict(), f"{save_path}/{model_save_name}")
    print('model saved')
    return model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc


def test_and_visualize_model(model, phase = 'test', num_images=4):
    """phase = 'train', 'valid', 'test'"""
    was_training = model.training
    model.eval()
    fig = plt.figure()

    running_loss, running_corrects, num_cnt = 0.0, 0, 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)  # batch의 평균 loss 출력

            running_loss    += loss.item() * inputs.size(0)
            running_corrects+= torch.sum(preds == labels.data)
            num_cnt += inputs.size(0)  # batch size

    #         if i == 2: break

        test_loss = running_loss / num_cnt
        test_acc  = running_corrects.double() / num_cnt       
        print('test done : loss/acc : %.2f / %.1f' % (test_loss, test_acc*100))

    # 예시 그림 plot
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)        

            # 예시 그림 plot
            for j in range(1, num_images+1):
                ax = plt.subplot(num_images//2, 2, j)
                ax.axis('off')
                ax.set_title('%s : %s -> %s'%(
                    'True' if class_names[str(labels[j].cpu().numpy())]==class_names[str(preds[j].cpu().numpy())] else 'False',
                    class_names[str(labels[j].cpu().numpy())],
                    class_names[str(preds[j].cpu().numpy())]
                    ))
                cv2.imshow(inputs.cpu().data[j])          
            if i == 0 : break

    model.train(mode=was_training);  # 다시 train모드로



if __name__ == "__main__":
    freeze_support()
    
    # 알파뱃순으로
    class_names = {
        "0": "apple",
        "1": "banana",
        "2": "carrot",
        "3": "cauliflower",
        "4": "garlic",
        "5": "ginger",
        "6": "grapes",
        "7": "kiwi",
        "8": "paprika",
        "9": "pineapple",
    }
    model_name = 'efficientnet-b0'  # b5
    model = EfficientNet.from_pretrained(model_name, num_classes=len(class_names))
    print(EfficientNet.get_image_size(model_name))

    batch_size  = 32
    random_seed = 555
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path   = f"{os.path.dirname(PROJECT_DIR)}/dataset/freshfood/all"
    save_path   = f"{os.path.dirname(PROJECT_DIR)}/saved_models/freshfood"


    ## make dataset
    mydataset = datasets.ImageFolder(
                                    data_path,
                                    transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ]))
    ## data split
    train_idx, tmp_idx = train_test_split(list(range(len(mydataset))), test_size=0.2, random_state=random_seed)
    datasets = {}
    datasets['train'] = Subset(mydataset, train_idx)
    tmp_dataset       = Subset(mydataset, tmp_idx)

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

    # ## 데이타 체크
    # num_show_img = 7
    # ## train check
    # inputs, classes = next(iter(dataloaders['train']))
    # out = torchvision.utils.make_grid(inputs[:num_show_img])  # batch의 이미지를 오려부친다
    # imshowt(out, title=[class_names[str(int(x))] for x in classes[:num_show_img]])
    # ## valid check
    # inputs, classes = next(iter(dataloaders['valid']))
    # out = torchvision.utils.make_grid(inputs[:num_show_img])
    # imshowt(out, title=[class_names[str(int(x))] for x in classes[:num_show_img]])
    # ## test check
    # inputs, classes = next(iter(dataloaders['test']))
    # out = torchvision.utils.make_grid(inputs[:num_show_img])
    # imshowt(out, title=[class_names[str(int(x))] for x in classes[:num_show_img]])
    # time.sleep(6)

    # print("fc 제외하고 freeze")
    # for n, p in model.named_parameters():
    #     if '_fc' not in n:
    #         p.requires_grad = False
    # model = torch.nn.parallel.DistributedDataParallel(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set gpu
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), 
                            lr = 0.05,
                            momentum=0.9,
                            weight_decay=1e-4)
    lmbda = lambda epoch: 0.98739
    exp_lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer_ft, lr_lambda=lmbda)

    model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc = train_model(
        save_path, model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)

    ## 결과 그래프 그리기
    print('best model : %d - %1.f / %.1f'%(best_idx, valid_acc[best_idx], valid_loss[best_idx]))
    fig, ax1 = plt.subplots()

    ax1.plot(train_acc, 'b-', label='train_acc')
    ax1.plot(valid_acc, 'r-', label='valid_acc')
    plt.plot(best_idx, valid_acc[best_idx], 'ro')
    ax1.set_xlabel('epoch')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('acc', color='k')
    ax1.tick_params('y', colors='k')

    ax2 = ax1.twinx()
    ax2.plot(train_loss, 'g-', label='train_loss')
    ax2.plot(valid_loss, 'k-', label='valid_loss')
    plt.plot(best_idx, valid_loss[best_idx], 'ro')
    ax2.set_ylabel('loss', color='k')
    ax2.tick_params('y', colors='k')

    fig.tight_layout()
    ax1.legend()
    ax2.legend()
    plt.show()

    # ## TEST!
    # print('학습완료')
    # test_and_visualize_model(model, phase = 'test')
