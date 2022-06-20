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
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        self.vid = MyVideoCapture(video_source)
        
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        # # 캡쳐 버튼
        # self.btn_snapshot = tkinter.Button(window, text='Snapshot', width=50, command = self.snapshot)
        # self.btn_snapshot.pack(anchor = tkinter.LEFT, expand=True)
                
        # 한번 호출되면, 업데이트 메소드가 매 밀리초마다 호출됨
        self.delay = 15
        self.update()

        self.window.mainloop()
        
    # def snapshot(self):
    #     ret, frame = self.vid.get_frame()
    #     if ret:
    #         cv2.imwrite()
        

    def update(self):
        ret, frame = self.vid.get_frame()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0,0,image=self.photo, anchor = tkinter.NW)

        self.window.after(self.delay, self.update)
        
        
        
class MyVideoCapture:
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("비디오 소스 열기 실패", video_source)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
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

    App(tkinter.Tk(), "Tkinter and OpenCV")

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
