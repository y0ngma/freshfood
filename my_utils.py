import cv2 # pip install opencv-python
import os, datetime, time


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

    save_dir = capture_vid(save_path)
    print(save_dir)