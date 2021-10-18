import os
import glob
from pathlib import Path
from posixpath import split


split_name_to_txt_path = {
    split_name: os.path.join(f"data/data/non_cropped/{split_name}.txt")
    for split_name in ["train", "test"]
}


cropped_dataset_root = "data/data/cropped/"
cropped_images_folder = os.path.join(cropped_dataset_root, "images")


all_images = glob.glob(os.path.join(
    cropped_images_folder, "*.png"))

all_images_name = [
    Path(path).stem
    for path in all_images
]


for split_name, split_txt in split_name_to_txt_path.items():
    print(split_name, split_txt)

    with open(split_txt, "rt") as f:
        split_prefixes = f.readlines()
        split_prefixes = [
            split_prefix.strip()
            for split_prefix in split_prefixes if split_prefix
        ]

        crops_for_this_split = []
        for prefix in split_prefixes:
            crops_for_this_split.extend(filter(
                lambda name: name.startswith(prefix),
                all_images_name
            ))
        
        with open(os.path.join(cropped_dataset_root, f"{split_name}.txt"), "wt") as new_split_file:
            new_split_file.write("\n".join(crops_for_this_split))
