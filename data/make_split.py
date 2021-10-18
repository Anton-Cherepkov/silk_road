from pathlib import Path
import random


dataset_folder = "data/data/non_cropped"
dst_train = "data/data/non_cropped/train.txt"
dst_test = "data/data/non_cropped/test.txt"
train_frac = 0.8 

names = [path.stem for path in (Path(dataset_folder) / "images").glob("*.tif")]
random.shuffle(names)
n = int(len(names) * train_frac)

with open(dst_train, "w") as f:
    f.write("\n".join(names[:n]))

with open(dst_test, "w") as f:
    f.write("\n".join(names[n:]))