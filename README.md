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
  ㄴ img_cap
    ㄴ captured_img.jpg
  train_main.py # 학습파일
  my_utils.py # 실행파일
```

## 학습
- 정확도 80% 목표
    - 97% 이상 달성
    - 다음 단계에서는 테스트결과 클래스별 보고서 작성(도표제시)
- train/eval 모드별 입력 이미지 해상도 제한없음
    - 1920*1080 FHD
    - 1280*720 HD
    - 640*480 SD
- 기본 UI에 구현
    - 1개의 상품 학습해서 캡쳐시 분류내용 출력 및 캡쳐

```bash
# 필요 라이브러리 설치 @ ~/freshfood/
pip install -r requirements.txt
python train_main.py
```
- 현재는 10개의 과일에 대하여 각 100장 정도로 학습한 모델이 첨부되어 있습니다.
  - apple, banana, carrot, cauliflower, garlic, ginger, grapes, paprika, pineapple, kiwi

## 분류기 실행
- 학습 후 새로이 저장된 모델의 경로를 인식하도록 코드 수정
- 웹캠이나 ipcamera 등을 구동하여 화면을 tkinter 윈도우에 송출합니다.
  - HD 이상 해상도 입력시 버튼조작을 위해 영상출력부 사이즈를 HD로 자동 조정
- 버튼을 누르면 캡쳐 후에 해당 이미지를 분류하고 그 결과를 출력합니다.
```bash
# 콘솔에서 신선상품 인식 실행하기
python my_utils.py
```