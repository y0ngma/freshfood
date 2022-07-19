# freshfood
## 데이터 및 라벨 준비
- 커스텀 데이터셋으로 인공지능 모델을 학습하려면 폴더명, 라벨을 수정합니다.
```bash
freshfood
  ㄴ sample
    ㄴ apple # 클래스
      ㄴ picture1.jpg
      ㄴ picture2.jpg
      ㄴ picture3.jpg
      ...
    ㄴ banana
    ...
    labels_map.txt # 라벨 정보
  ㄴ saved_models
    ㄴ 20220627_epoch@9.pt # 학습된 모델
  train_main.py # 학습파일
  my_utils.py # 실행파일
```

## 학습 후 저장된 모델의 경로를 인식하도록 코드 수정
```bash
# 필요 라이브러리 설치 @ ~/freshfood/
pip install -r requirements.txt
python train_main.py
```

## 분류기 실행
- 웹캠이나 ipcamera 등을 구동하여 화면을 tkinter 윈도우에 송출합니다. 
- 버튼을 누르면 캡쳐 후에 해당 이미지를 분류하고 그 결과를 출력합니다.
```bash
# 콘솔에서 신선상품 인식 실행하기
python my_utils.py
```