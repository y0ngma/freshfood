import cv2 # pip install opencv-python
import os, datetime, time
from tkinter import *


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

if __name__ == "__main__":
    PROJECT_DIR    = os.path.dirname(os.path.abspath(__file__))
    save_path      = f"{PROJECT_DIR}\img_cap"
    if not os.path.isdir(save_path): os.makedirs(save_path)

    # save_dir = capture_vid(save_path)
    # print(save_dir)
    
    root = Tk()
    root.title("딥러닝 기반의 신선상품 인식기술 개발")
    root.geometry("640x480+100+100")
    
    root.resizable(False, False)
    
    def btncmd():
        print('clicked')

    Label(root, text="신선제품을 저울에 올리세요").pack(side="top")    
    vid_frame = Frame(root, relief="solid", bd=3)
    vid_frame.pack(side='left', expand=True)
    Label(vid_frame, text='보기').pack()
    # pic = PhotoImage(file="")
    # command=capture_vid(save_path)
    
    btn_frame = LabelFrame(root, relief="solid", bd=1, text="버튼")
    btn_frame.pack(side='right', fill='both')
    btn1 = Button(btn_frame, padx=50, pady=20, fg='blue', bg='pink', command=btncmd, text="탐지")
    btn1.pack(side='bottom')
    Button(btn_frame, padx=50, pady=20, text="다시탐지").pack()
    Button(btn_frame, padx=50, pady=20, text="항목 직접입력").pack()

    Button(root, padx=50, pady=20, bg="grey", text="점장호출").pack(side="bottom")
    root.mainloop()
    