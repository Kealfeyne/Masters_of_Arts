import pandas as pd
from tqdm import tqdm
from PIL import Image


path_to_data = "data/images/"  # processed_train/ Аугментированный датасет

train_labels = pd.read_csv("data/train.csv", sep=",")  # processed_train Метки аугментированного датасета

for image_name in tqdm(train_labels['image'].values):
    try:
        image = Image.open(path_to_data + image_name).convert("RGB")
    except:
        print(image_name)

# image = Image.open(path_to_train_data + "d2570d7e31fb48ff86f4b42d5d0bd1b2.jpeg").convert("RGB")