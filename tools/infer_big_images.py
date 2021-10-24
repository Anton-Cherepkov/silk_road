# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import glob
import os
from pathlib import Path
from tqdm.auto import tqdm

import mmcv
from mmcv.runner import wrap_fp16_model
import cv2
import numpy as np

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette


def get_args():
    parser = ArgumentParser()
    parser.add_argument('input_folder', help='Folder with big images')
    parser.add_argument('output_folder', help="Folder to save masks & visualization")
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--fp16', action="store_true")
    parser.add_argument("--start_ix", type=int)
    parser.add_argument("--end_ix", type=int)
    args = parser.parse_args()
    return args


def draw_mask(img, mask, color=(0, 0, 255)):
    if isinstance(img, str):
        img = cv2.imread(img)  # (h, w, 3)
        assert img.ndim == 3
        assert img.shape[-1] == 3

    assert 0 <= mask.min() <= mask.max() <= 1

    bgr_mask = np.zeros_like(img)
    bgr_mask[:, :] = color
    bgr_mask = bgr_mask * mask[..., None]
    bgr_mask = bgr_mask.astype(img.dtype)

    img_with_mask = cv2.addWeighted(img, 0.65, bgr_mask, 0.35, 0)
    return img_with_mask


def get_tif_images_in_folder(folder: str):
    tif_imgs =  sorted(list(glob.glob(os.path.join(folder, "*.tif"))))
    print(f"Found {len(tif_imgs)} tif images")
    return tif_imgs


def main():
    args = get_args()

    # build the model from a config file and a checkpoint file
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.test_cfg['mode'] = 'slide'
    cfg.model.test_cfg['stride'] = (200, 200)
    cfg.model.test_cfg['crop_size'] = (512, 512)
    model = init_segmentor(cfg, args.checkpoint, device=args.device)
    if args.fp16:
        wrap_fp16_model(model)
        print("FP16 mode ON")

    # output folders
    visualization_folder = os.path.join(
        args.output_folder,
        "visualization",
    )
    masks_folder = os.path.join(
        args.output_folder,
        "masks",
    )
    for folder in [visualization_folder, masks_folder]:
        os.makedirs(folder, exist_ok=True)

    # inference
    imgs = list(get_tif_images_in_folder(args.input_folder))
    if args.start_ix is not None:
        assert args.end_ix is not None
        imgs = imgs[args.start_ix:args.end_ix]
        print(f"{len(imgs)} images will be processed")

    for image in tqdm(imgs):
        mask = inference_segmentor(model, image)[0]
        img_with_mask = draw_mask(image, mask)

        image_name = Path(image).stem
        visualization_path = os.path.join(
            visualization_folder, f"{image_name}.png"
        )
        mask_path = os.path.join(
            masks_folder, f"{image_name}.npy"
        )
        
        cv2.imwrite(visualization_path, img_with_mask)
        np.save(mask_path, mask)


if __name__ == '__main__':
    main()