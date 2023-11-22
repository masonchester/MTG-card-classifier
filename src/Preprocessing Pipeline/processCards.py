import json

with open("scryfall_default_cards.json", 'r', encoding='utf-8') as f:
    cards = json.load(f)

processed_cards_dict = {}
types = ['Creature', 'Instant', 'Sorcery', 'Enchantment', 'Artifact', 'Land']

for card in cards:
    if 'image_uris' in card and 'art_crop' in card['image_uris'] and 'type_line' in card:
        card_name = card['name']
        primary_type = card['type_line'].split("—")[0].strip()
        highres = card.get('highres_image', False)
        
        if primary_type in types:
             if card_name not in processed_cards_dict or (card_name in processed_cards_dict and highres):
                processed_card = {
                    'name': card_name,
                    'image_url': card['image_uris']['art_crop'],
                    'type': primary_type,
                    'highres_image': highres
                }
                processed_cards_dict[card_name] = processed_card

processed_cards = list(processed_cards_dict.values())

with open("processed_cards.json", 'w', encoding='utf-8') as f:
    json.dump(processed_cards, f, indent=4)

print(f"Saved {len(processed_cards)} unique cards to processed_cards.json")
