import os, glob, shutil
from PIL import Image
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
data_path   = f"{os.path.dirname(PROJECT_DIR)}/dataset/{os.path.basename(PROJECT_DIR)}"
one_folder  = "C:/home/dataset/freshfood/all"
classes     = [#"beetroot","bell pepper","cabbage","capsicum","chilli pepper","corn","cucumber","eggplant","jalepeno","kiwi","lemon","lettuce","onion","orange","paprika","pear","peas","pomegranate","potato","raddish","soy beans","spinach","sweetcorn","sweetpotato","tomato","turnip",
"apple","banana","carrot","cauliflower","garlic","ginger","grapes","mango","pineapple","watermelon",
]

def toRGBA(file):
    img = Image.open(file).convert('RGBA')
    x = np.array(img)
    r, g, b, a = np.rollaxis(x, axis = -1)
    r[a == 0] = 255
    g[a == 0] = 255
    b[a == 0] = 255
    x = np.dstack([r, g, b, a])
    img = Image.fromarray(x, 'RGBA')
    rgb_im = img.convert("RGB")
    
    return rgb_im

for cls in classes:
    if not os.path.isdir(f"{one_folder}/{cls}"): os.makedirs(f"{one_folder}/{cls}")
    pics = glob.glob(f"{data_path}/**/{cls}/*", recursive=True)
    cnt = 0
    for idx, pic in enumerate(pics):
        ext = os.path.splitext(pic)[1]
        if ext == ".png":
            img = toRGBA(pic)
            # print(f"변환되어 저장함 {one_folder}/{cls}/{idx}.jpg")
            img.save(f"{one_folder}/{cls}/{idx}.jpg")
        else:
            dst = f"{one_folder}/{cls}/{idx}{ext}"
            shutil.copy(pic, dst)
        cnt += 1
    print("클래스명:{:<15} 기존파일수:{:<5} 옮긴수:{:<5}".format(cls, len(pics), cnt))