import os, glob, shutil

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
data_path   = f"{os.path.dirname(PROJECT_DIR)}/dataset/{os.path.basename(PROJECT_DIR)}"
one_folder  = "C:/home/dataset/freshfood/all"
classes     = [#"beetroot","bell pepper","cabbage","capsicum","chilli pepper","corn","cucumber","eggplant","jalepeno","kiwi","lemon","lettuce","onion","orange","paprika","pear","peas","pomegranate","potato","raddish","soy beans","spinach","sweetcorn","sweetpotato","tomato","turnip",
"apple","banana","carrot","cauliflower","garlic","ginger","grapes","mango","pineapple","watermelon",
]
for cls in classes:
    if not os.path.isdir(f"{one_folder}/{cls}"): os.makedirs(f"{one_folder}/{cls}")
    pics = glob.glob(f"{data_path}/**/{cls}/*", recursive=True)
    for idx, pic in enumerate(pics):
        ext = os.path.splitext(pic)[1]
        dst = f"{one_folder}/{cls}/{idx}{ext}"
        shutil.move(pic, dst)
    print(cls, len(pics))
# for cls in classes: # 옮긴 갯수 동일 검증
#     print(cls, len(os.listdir(f"{one_folder}/{cls}")))