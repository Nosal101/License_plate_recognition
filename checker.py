import json

# Wczytaj plik JSON
with open('output.json', 'r') as json_file:
    data = json.load(json_file)

# Przejrzyj każdą parę klucz-wartość w słowniku
max_points = 0
earned = 0
for key, value in data.items():
    # Upewnij się, że wartość jest stringiem
    if not isinstance(value, str):
        print(f'Value for key {key} is not a string. Skipping this key-value pair.')
        continue

    # Porównaj każdy znak klucza z wartością
    same_char_count = sum(k_char == v_char for k_char, v_char in zip(key, value))
    if len(key) == 7:
        if same_char_count == 7:
            earned += 10
        else:
            earned += same_char_count
        max_points += 10
    elif len(key) == 8:
        if same_char_count == 8:
            earned += 11
        else:
            earned += same_char_count
        max_points += 11
print(f'You earned {earned} out of {max_points} points. That is {earned/max_points*100:.2f}%.')