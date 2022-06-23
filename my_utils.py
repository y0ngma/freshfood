import cv2 # pip install opencv-python
import os, datetime, time
from tkinter import *
from PIL import Image, ImageTk
import tkinter
import threading

import json
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

def capture_vid(
    save_path: str,
    location: str="office",
    src_address: str="rtsp://admin:neuro1203!@192.168.0.73:554/ISAPI/streaming/channels/101"
    ):
    """
    ip카메라 캡쳐
    
    """
    frameRate      = 30
    font           = cv2.FONT_HERSHEY_SIMPLEX
    org            = (1,25)
    size, BaseLine = cv2.getTextSize(location, font, 1, 2)

    try:
        cap = cv2.VideoCapture(src_address)
    except:
        print('외부 영상 소스가 없습니다')
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # direct show

    if cap.isOpened():
        print('width: {}, height: {}'.format(cap.get(3), cap.get(4)))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret==False: break # 프레임정보 return 실패시
        # cv2.circle(frame, org, 3, (255,0,255), 2)
        # cv2.rectangle(frame, org, (org[0]+size[0], org[1]-size[1]), (0,0,255))
        cv2.putText(frame, location, org, font, 1, (255,0,0), 2)
        cv2.imshow(location, frame)

        key = cv2.waitKey(frameRate)
        if key == 27: # esc버튼 누르면
            stamp    = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            save_dir = f"{save_path}\{location}_{stamp}.png"
            cv2.IMREAD_UNCHANGED
            cv2.imwrite(save_dir, frame)
            print(f"저장경로 {save_dir}")
            time.sleep(1)
            break
    if cap.isOpened(): cap.release()
    cv2.destroyAllWindows()
    
    return save_dir


# def camthread():
#     color = []
#     cap = cv2.VideoCapture("rtsp://admin:neuro1203!@192.168.0.73:554/ISAPI/streaming/channels/101")
#     panel = None
    
#     if (cap.isOpened()==False):
#         print("Unable to read camera feed")
        
#     while True:
#         ret, color = cap.read()
#         if (color != []):
#             # cv2.imshow('uvc', color)
#             image = cv2.resize(color, (100,100))
#             image = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
#             image = Image.fromarray(image)
#             image = ImageTk.PhotoImage(image)
            
#             if panel is None:
#                 panel = tk.Label(image=image)
#                 panel.image = image
#                 panel.pack(side='left')
#             else:
#                 panel.configure(image=image)
#                 panel.image = image

#             cv2.waitKey(1)

import PIL.Image, PIL.ImageTk
class App:

    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.vid = MyVideoCapture(video_source)

        location       ='office'
        PROJECT_DIR    = os.path.dirname(os.path.abspath(__file__))
        self.snapshot_dir = f"{PROJECT_DIR}/img_cap"
        label_dir     = f'{PROJECT_DIR}/img_cap/labels_map.txt'
        model_path    = f"{os.path.dirname(PROJECT_DIR)}/saved_models/freshfood"

        # Load ImageNet class names
        labels_map = json.load(open(label_dir))
        self.labels_map = [labels_map[str(i)] for i in range(len(labels_map))]

        ## 모델 로드
        model_name = 'efficientnet-b0'  # b5
        self.model = EfficientNet.from_pretrained(model_name, num_classes=10)
        self.model.load_state_dict(torch.load(f"{model_path}/20220617_epoch@4.pt"))
        self.model.eval()

        self.tfms = transforms.Compose(
            [transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])


        ## 영상폭+GUI패널폭 여유를 줘서 윈도우 폭 설정(영상이 너무 큰 경우 줄임)
        if (self.vid.width > 1280)|(self.vid.height > 720):
            canvas_w, canvas_h = 1280, 720
        else:
            canvas_w, canvas_h = self.vid.width, self.vid.height

        Label(window, text="신선제품을 저울에 올리세요").pack(side="top")
        GUI_pad = 200
        window.geometry(f'{int(canvas_w+GUI_pad)}x{int(canvas_h)}+0+0') # 가로세로 크기 및 시작위치
        # window.resizable(False, False) # 가로세로 변경 못하게

        ## 영상표시부
        vid_frame   = Frame(window, relief="solid", bd=3)
        self.canvas = tkinter.Canvas(vid_frame, width=canvas_w, height=canvas_h)
        vid_frame.pack(side="left", expand=True)
        self.canvas.pack()

        ## GUI부 틀
        gui_frame = Frame(window, relief="solid", bd=0)
        gui_frame.pack(side="right", expand=True)
        ## 탐지 항목 확인부
        msg_frame = LabelFrame(gui_frame, relief="solid", bd=1, text="품목명을 확인하세요")
        msg_frame.pack(side="top")
        Label(msg_frame, text="{: <19}{:>6}".format('항목명', '정확도')).pack()
        # msg_frame.pack(side="top", fill='both')
        item_var = StringVar()
        self.btn_item1 = Radiobutton(msg_frame, text=f"apple", value="apple", variable=item_var)
        self.btn_item1.select()
        self.btn_item2 = Radiobutton(msg_frame, text="바나나", value="banana", variable=item_var)
        self.btn_item3 = Radiobutton(msg_frame, text="가지", value="eggplant", variable=item_var)
        self.btn_item4 = Radiobutton(msg_frame, text="파인애플", value="pineapple", variable=item_var)
        self.btn_item5 = Entry(msg_frame, width=80)
        # btn_item5 = Text(msg_frame, width=80, height=10)
        self.btn_item1.pack()
        self.btn_item2.pack()
        self.btn_item3.pack()
        self.btn_item4.pack()
        self.btn_item5.pack()
        self.btn_item5.insert(0, # 문구 입력 위치 (row, col)
                         "품목이 없는 경우 입력하세요")

        def get_var():
            """수정/확인 한 품목명을 가져와서 품목별 단가와 무게를 곱하여 가격 도출"""
            Ent = self.btn_item5.get()
            if not Ent == "품목이 없는 경우 입력하세요": print(Ent)
            else: print(item_var.get())

        btn_get = Button(msg_frame, text="수정/확인", command=get_var)
        btn_get.pack()

        ## 버튼부
        btn_frame = LabelFrame(gui_frame, relief="solid", bd=1, text="버튼을 누르세요")
        btn_frame.pack(side="bottom")

        self.btn_snapshot = tkinter.Button(
            btn_frame, padx=80, pady=20, text='탐지', fg='blue', bg='pink', command = self.whatis)
        # self.btn_snapshot.pack(anchor = tkinter.CENTER, expand=True)
        self.btn_snapshot.pack(side="top")
        Button(btn_frame, padx=80, pady=20, text="캡쳐", command = self.snapshot).pack()
        Button(btn_frame, padx=80, pady=20, text="무게 측정").pack()
        Button(btn_frame, padx=80, pady=20, text="가격표 발행").pack()
        Button(btn_frame, padx=80, pady=20, text="점장호출", bg="grey").pack()

        ## 결과값 불러오기
        # self.results = self.whatis()
        
        # btn_item1 = Radiobutton(msg_frame, text=f"{self.results[0][0]:<10} {self.results[0][1]:>6.2f}", value=self.results[0][0], variable=item_var)

        ## 한번 호출되면, 업데이트 메소드가 매 밀리초마다 호출됨
        self.delay = 15
        self.update()

        self.window.mainloop()

    
    def changetxt(self):
        self.btn_item1.configure(text=f"{self.results[0][0]:<10} {self.results[0][1]:>6.2f}")

    def snapshot(self):
        ret, frame = self.vid.get_frame()
        if ret:
            self.snapshot_file = f'{self.snapshot_dir}\{time.strftime("%Y-%m-%d %H-%M-%S")}.jpg'
            cv2.imwrite(self.snapshot_file, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            return self.snapshot_file

    
    def whatis(self):
        """사진한장에 대해 탐지결과를 반환"""
        self.save_dir = self.snapshot()

        ## Preprocess image
        img = Image.open(self.save_dir)
        img = self.tfms(img).unsqueeze(0)

        # Classify
        start_time = datetime.datetime.now() # 시간측정
        with torch.no_grad():
            outputs = self.model(img)
        # Print predictions
        self.results = list()
        print('---------')
        for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
            prob = torch.softmax(outputs, dim=1)[0, idx].item()
            print('{label:<75} ({p:0>6.2f}%)'.format(label=self.labels_map[idx], p=prob*100))
            self.results.append((self.labels_map[idx], prob*100))
        print("소요시간=> ", datetime.datetime.now() - start_time)

        ## 분류된항목별 정확도로 버튼텍스트 수정
        # self.changetxt()
        self.btn_item1.configure(text="{: <19}{:>6.2f}%".format(self.results[0][0], self.results[0][1]))
        self.btn_item2.configure(text="{: <19}{:>6.2f}%".format(self.results[1][0], self.results[1][1]))
        self.btn_item3.configure(text="{: <19}{:>6.2f}%".format(self.results[2][0], self.results[2][1]))
        self.btn_item4.configure(text="{: <19}{:>6.2f}%".format(self.results[3][0], self.results[3][1]))

        return self.results


    def update(self):
        ret, frame = self.vid.get_frame()
        if ret:
            image=PIL.Image.fromarray(frame)
            if (image.width > 1280)|(image.height > 720):
                image = image.resize((1280,720), PIL.Image.NEAREST)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0,0,image=self.photo, anchor = tkinter.NW)
        self.window.after(self.delay, self.update)



class MyVideoCapture:

    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("비디오 소스 열기 실패", video_source)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print("original size of width:{} height:{}".format(self.width, self.height))


    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

        
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()



if __name__ == "__main__":
    # PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    # model_path  = f"{os.path.dirname(PROJECT_DIR)}/saved_models/freshfood"
    # save_path   = f"{PROJECT_DIR}/img_cap"
    # label_dir   = f'{save_path}/labels_map.txt'
    # if not os.path.isdir(save_path): os.makedirs(save_path)

    # App(tkinter.Tk(), "Tkinter and OpenCV", video_source="rtsp://admin:neuro1203!@192.168.0.73:554/ISAPI/streaming/channels/101")
    App(tkinter.Tk(), "Tkinter and OpenCV")