import os
import requests
import json
from tqdm import tqdm

with open("processed_cards.json", 'r', encoding='utf-8') as f:
    cards = json.load(f)

os.makedirs("images/train", exist_ok=True)
os.makedirs("images/validation", exist_ok=True)
os.makedirs("images/test", exist_ok=True)

def download_image(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

train_size = int(0.7 * len(cards))
val_size = int(0.15 * len(cards))

for i, card in tqdm(enumerate(cards), total=len(cards), desc="Downloading images"):
    img_url = card['image_url']
    img_name = os.path.basename(img_url.split("?")[0])
    if i < train_size:
        download_image(img_url, os.path.join("images/train", img_name))
    elif i < train_size + val_size:
        download_image(img_url, os.path.join("images/validation", img_name))
    else:
        download_image(img_url, os.path.join("images/test", img_name))
