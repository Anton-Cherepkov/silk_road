import json


with open("positive_pixels_cnt.json", "r") as f:
    mapping = json.load(f)
    name_cnt = list(mapping.items())
    name_cnt = list(sorted(name_cnt, key=lambda pair: pair[1], reverse=True))

with open("positive_pixels_cnt.txt", "wt") as f:
    for name, cnt in name_cnt:
        f.write(f"{name}\t{cnt}\n")

with open("names_with_positive_pixels.txt", "wt") as f:
    for name, cnt in name_cnt:
        if cnt > 0:
            f.write(f"{name}\n")
