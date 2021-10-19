import json


json_path = "positive_pixels_cnt.json"

with open(json_path, "r") as f:
    name_to_positive_pixels_cnt = json.load(f)


print(f"{len(name_to_positive_pixels_cnt)} crops...")

num_with_positive_pixels_more_zero = len(list(filter(
    lambda cnt: cnt > 0,
    name_to_positive_pixels_cnt.values(),
)))
print(f"{num_with_positive_pixels_more_zero} crops with >0 positive pixels")


# 127584 crops...
# 3507 crops with >0 positive pixels
