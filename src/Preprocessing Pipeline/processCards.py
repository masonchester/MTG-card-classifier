import json

with open("scryfall_default_cards.json", 'r', encoding='utf-8') as f:
    cards = json.load(f)

processed_cards = []
unique_card_names = set()
types = ['Planeswalker', 'Creature', 'Instant', 'Sorcery', 'Enchantment', 'Artifact', 'Land']

for card in cards:
    if 'image_uris' in card and 'art_crop' in card['image_uris'] and 'type_line' in card and 'name' in card:
        card_name = card['name']
        if card_name not in unique_card_names:
            card_colors = card.get('colors', [])
            if len(card_colors) == 0:
                card_colors = ["C"]
            if len(card_colors) == 1:
                primary_type = card['type_line'].split("—")[0].strip()
                if primary_type in types:
                    processed_card = {
                        'name': card_name,
                        'image_url': card['image_uris']['art_crop'],
                        'colors': card_colors,
                        'type': primary_type
                    }
                    processed_cards.append(processed_card)
                    unique_card_names.add(card_name)

with open("processed_cards.json", 'w', encoding='utf-8') as f:
    json.dump(processed_cards, f, indent=4)

print(f"Saved {len(processed_cards)} UNIQUE!!!!!! cards to processed_cards.json")
