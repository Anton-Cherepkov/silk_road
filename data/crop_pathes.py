import argparse
from pathlib import Path
from tqdm import tqdm
import cv2


def do_crop(image_path, dst_folder, window_side, stride):
    image = cv2.imread(str(image_path))

    h, w = image.shape[:2]
    assert (h - window_side) % stride == (w - window_side) % stride == 0

    image_name = Path(image_path).stem
    dst_folder = Path(dst_folder)

    for x in range(0, w - window_side + 1, stride):
        for y in range(0, h - window_side + 1, stride):
            crop = image[y:y+window_side, x:x+window_side]
            assert crop.shape[:2] == (window_side, window_side)

            cv2.imwrite(str(dst_folder / f"{image_name}_{x}_{y}.png"), crop)



def main(args):
    with open(args.names) as f:
        names = f.read().strip().split("\n")
    
    src_dataset_folder = Path(args.src_dataset_folder)
    dst_dataset_folder = Path(args.dst_dataset_folder)

    labels = [folder.name for folder in (src_dataset_folder / "masks").iterdir()]

    (dst_dataset_folder / "images").mkdir(parents=True)
    for label in labels:
        (dst_dataset_folder / "masks" / label).mkdir(parents=True)

    for name in tqdm(names):
        png_name = name + ".png"
        tif_name = name + ".tif"
        do_crop(
            src_dataset_folder / "images" / tif_name, 
            dst_dataset_folder / "images",
            args.window_size,
            args.stride
        )

        for label in labels:
            do_crop(
                src_dataset_folder / "masks" / label / png_name, 
                dst_dataset_folder / "masks" / label,
                args.window_size,
                args.stride
            )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--names")
    parser.add_argument("--src_dataset_folder")
    parser.add_argument("--dst_dataset_folder")
    parser.add_argument("--window_size", type=int)
    parser.add_argument("--stride", type=int)

    main(parser.parse_args())
    