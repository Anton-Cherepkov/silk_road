import glob
import os
import cv2
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from typing import Dict
import json


cropped_masks_folder = "data/data/cropped/masks/road"
name_to_positive_pixels: Dict[str, int] = dict()

for mask_path in tqdm(glob.glob(os.path.join(cropped_masks_folder, "*.png"))):
    mask = cv2.imread(mask_path)
    mask = mask[:, :, 0]
    name = Path(mask_path).stem

    positive_pixels_cnt = int((mask == 1).sum())
    assert (mask == 0).sum() == np.prod(mask.shape) - positive_pixels_cnt

    name_to_positive_pixels[name] = positive_pixels_cnt

with open("positive_pixels_cnt.json", "w") as f:
    json.dump(name_to_positive_pixels, f)
