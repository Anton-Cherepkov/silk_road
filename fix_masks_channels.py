import cv2
import numpy as np
import glob
from tqdm.auto import tqdm


def get_all_masks_paths():
    return glob.glob("data/data/cropped/masks/**/*.png")

fixed_cnt = 0

for path in tqdm(get_all_masks_paths()):
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
        cv2.imwrite(path, mask)
        fixed_cnt += 1

print(f"Fixed {fixed_cnt} mask(s)")
