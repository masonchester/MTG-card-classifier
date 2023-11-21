import os
import requests
import json
from tqdm import tqdm

def create_directories(base_path, types):
    for type_ in types:
        dir_path = os.path.join(base_path, type_)
        os.makedirs(dir_path, exist_ok=True)

def download_image(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)

with open("processed_cards.json", 'r', encoding='utf-8') as f:
    cards = json.load(f)

types = [ 'Creature', 'Instant', 'Sorcery', 'Enchantment', 'Artifact', 'Land']

base_paths = {
    "train": "images/train",
    "validation": "images/validation",
    "test": "images/test"
}

for path in base_paths.values():
    create_directories(path, types)

train_size = int(0.7 * len(cards))
val_size = int(0.15 * len(cards))

for i, card in tqdm(enumerate(cards), total=len(cards), desc="Downloading images"):
    img_url = card['image_url']
    img_name = os.path.basename(img_url.split("?")[0])
    type_ = card['type']
    if type_ not in types:
        continue 
    directory = "train" if i < train_size else "validation" if i < train_size + val_size else "test"
    type_directory = os.path.join(base_paths[directory], type_)
    
    os.makedirs(type_directory, exist_ok=True)
    
    save_path = os.path.join(type_directory, img_name)
    if not os.path.exists(save_path):
        download_image(img_url, save_path)
