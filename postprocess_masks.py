import argparse
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
from postprocessing.process import img_to_ske_G, nx2polyline, read_tfw_file, polyline2shapefile, zip_and_remove, draw_mask, draw_polyline
from postprocessing.tfw import read_tfw_file


def main(args):
    imgs_folder = Path(args.imgs_folder)
    output_folder = Path(args.output_folder)
    
    output_folder.mkdir(exist_ok=True, parents=True)
    
    for mask_path in tqdm(list(Path(args.masks_folder).glob("*.npy"))):
        mask = np.load(mask_path)
        tfw = read_tfw_file(imgs_folder / (mask_path.stem + ".tfw"))
        _, _, G = img_to_ske_G(mask)
        polyline = nx2polyline(G)
        
        shapefile_dst_folder = str(output_folder / mask_path.stem)
        
        polyline2shapefile(polyline, shapefile_dst_folder, tfw)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks_folder")
    parser.add_argument("--output_folder")
    parser.add_argument("--imgs_folder")
    
    main(parser.parse_args())