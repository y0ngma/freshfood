import cv2 # pip install opencv-python
import os, datetime, time
from tkinter import *
from PIL import Image, ImageTk
import tkinter
import threading

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

    def __init__(self, window, window_title, save_path='./', video_source=0):
        self.window = window
        self.window.title(window_title)
        Label(window, text="신선제품을 저울에 올리세요").pack(side="top")
        
        ## 영상표시부
        vid_frame = Frame(window, relief="solid", bd=3)
        vid_frame.pack(side="left", expand=True)

        self.video_source = video_source
        self.vid = MyVideoCapture(video_source)

        if (self.vid.width > 1280)|(self.vid.height > 720):
            self.canvas = tkinter.Canvas(vid_frame, width = 1280, height = 720)
        else:
            self.canvas = tkinter.Canvas(vid_frame, width = self.vid.width, height = self.vid.height)
    
        self.canvas.pack()

        ## GUI부 틀
        gui_frame = Frame(window, relief="solid", bd=0)
        gui_frame.pack(side="right", expand=True)
        ## 메세지부
        msg_frame = LabelFrame(gui_frame, relief="solid", bd=1, text="품목명을 확인하세요")
        msg_frame.pack(side="top", fill='both')
        
        item_var = StringVar()
        btn_item1 = Radiobutton(msg_frame, text="사과", value="apple", variable=item_var)
        btn_item1.select()
        btn_item2 = Radiobutton(msg_frame, text="바나나", value="banana", variable=item_var)
        btn_item3 = Radiobutton(msg_frame, text="가지", value="eggplant", variable=item_var)
        btn_item4 = Radiobutton(msg_frame, text="파인애플", value="pineapple", variable=item_var)
        btn_item1.pack()
        btn_item2.pack()
        btn_item3.pack()
        btn_item4.pack()
        # btn_item5 = Text(msg_frame, width=80, height=10)
        btn_item5 = Entry(msg_frame, width=80)
        btn_item5.insert(0,"품목이 없는 경우 입력하세요")
        btn_item5.pack()
        
        def get_var():
            """수정/확인 한 품목명을 가져와서 품목별 단가와 무게를 곱하여 가격 도출"""
            Ent = btn_item5.get()
            if not Ent == "품목이 없는 경우 입력하세요": print(Ent)
            else: print(item_var.get())

        btn_get = Button(msg_frame, text="수정/확인", command=get_var)
        btn_get.pack()

        ## 버튼부
        btn_frame = LabelFrame(gui_frame, relief="solid", bd=1, text="버튼을 누르세요")
        btn_frame.pack(side="bottom", fill="both")
        
        self.btn_snapshot = tkinter.Button(
            btn_frame, padx=80, pady=20, text='탐지', fg='blue', bg='pink', command = self.snapshot(save_path))
        # self.btn_snapshot.pack(anchor = tkinter.CENTER, expand=True)
        self.btn_snapshot.pack(side="top")
        Button(btn_frame, padx=80, pady=20, text="항목 직접입력").pack()
        Button(btn_frame, padx=80, pady=20, text="무게 측정").pack()
        Button(btn_frame, padx=80, pady=20, text="가격표 발행").pack()
        Button(btn_frame, padx=80, pady=20, text="점장호출", bg="grey").pack()

        ## 한번 호출되면, 업데이트 메소드가 매 밀리초마다 호출됨
        self.delay = 15
        self.update()

        self.window.mainloop()

        
    def snapshot(self, save_path, location='office'):
        ret, frame = self.vid.get_frame()
        if ret:
            save_dir = f'{save_path}\{location}_{time.strftime("%Y-%m-%d %H-%M-%S")}.jpg'
            cv2.imwrite(save_dir, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


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
    PROJECT_DIR    = os.path.dirname(os.path.abspath(__file__))
    save_path      = f"{PROJECT_DIR}\img_cap"
    if not os.path.isdir(save_path): os.makedirs(save_path)

    App(tkinter.Tk(), "Tkinter and OpenCV", save_path+"/", video_source="rtsp://admin:neuro1203!@192.168.0.73:554/ISAPI/streaming/channels/101")
    # App(tkinter.Tk(), "Tkinter and OpenCV", save_path+"/", video_source=0)

    # # save_dir = capture_vid(save_path)
    # # print(save_dir)
    
    # def btncmd():
    #     print('clicked')

    # root = Tk()
    # width, height = 920, 640
    # # root.geometry(f"{width}x{height}+{1920-width}+{1080-height}") # 가로세로 크기 및 시작위치
    # root.geometry(f"{width}x{height}+0+0") # 가로세로 크기 및 시작위치
    # root.resizable(False, False) # 가로세로 변경 못하게
    # root.title("딥러닝 기반의 신선상품 인식기술 개발") # 창 이름
    # Label(root, text="신선제품을 저울에 올리세요").pack(side="top")    

    # ## 영상표시부
    # vid_frame = Frame(root, relief="solid", bd=3)
    # vid_frame.pack(side='left', expand=True)
    # # command=capture_vid(save_path)
    # pic = PhotoImage(file="C:/home/freshfood/img_cap/banana.png") # jpg는 안됨
    # Label(vid_frame, image=pic).pack(side="left")
    
    
    # # thread_img = threading.Thread(target=camthread, args=())
    # # thread_img.deamon = True
    # # thread_img.start()
    

    # ## 버튼부분
    # btn_frame = LabelFrame(root, relief="solid", bd=1, text="버튼")
    # btn_frame.pack(side='right', fill='both')

    # btn1 = Button(btn_frame, padx=50, pady=20, fg='blue', bg='pink', command=btncmd, text="탐지")
    # btn1.pack(side='bottom')
    # Button(btn_frame, padx=50, pady=20, text="다시탐지").pack()
    # Button(btn_frame, padx=50, pady=20, text="항목 직접입력").pack()

    # Button(root, padx=50, pady=20, bg="grey", text="점장호출").pack(side="bottom")

    # root.mainloop()
