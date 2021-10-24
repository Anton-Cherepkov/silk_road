# way to upload img
# func to make a prediction
# funcrtiob  to make prewdictio n on image
# functio to show the resulting image
# function to generate geo format
import os
from flask import request
from flask import Flask
from flask import render_template
from flask import send_file

from typing import Optional

from argparse import ArgumentParser
import glob
import os
from pathlib import Path
from tqdm.auto import tqdm
from dataclasses import dataclass
# import sns
import mmcv
from mmcv.runner import wrap_fp16_model
import cv2
import numpy as np

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette


FP_16_MODE = None

PREDICTIONS_OUTPUT_FOLDER = "static/predictions"
UPLOAD_FOLDER = 'static/uploads'


@dataclass
class PredictionInformation:
    visualization_path: str
    mask_path: str
    shapefile_path: Optional[str]


def get_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--fp16', type=bool, default=False)
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
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
    return glob.glob(os.path.join(folder, "*.tif"))


def init_segmentation_model():
    # build the model from a config file and a checkpoint file
    args = get_args()
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.test_cfg['mode'] = 'slide'
    cfg.model.test_cfg['stride'] = (200, 200)
    cfg.model.test_cfg['crop_size'] = (512, 512)
    model = init_segmentor(cfg, args.checkpoint, device=args.device)
    global fp16_mode
    if args.fp16:
        wrap_fp16_model(model)
        fp16_mode = "ON"
    else:
        fp16_mode = "OFF"
    return model


def create_folders():
    masks_folder = os.path.join(PREDICTIONS_OUTPUT_FOLDER, "masks")
    visualizations_folder = os.path.join(PREDICTIONS_OUTPUT_FOLDER, "visualization")
    for folder in [masks_folder, visualizations_folder, UPLOAD_FOLDER]:
        os.makedirs(folder, exist_ok=True)


def predict(image, model, tfw_path: Optional[str]) -> PredictionInformation:
    mask = inference_segmentor(model, image)[0]
    visualization = draw_mask(image, mask)

    image_name = Path(image).stem

    prediction_info = PredictionInformation(
        visualization_path=os.path.join("visualization", f"{image_name}.jpg"),
        mask_path=os.path.join("masks", f"{image_name}.npy"),
        shapefile_path=None if not tfw_path else "debug",
    )

    np.save(os.path.join(PREDICTIONS_OUTPUT_FOLDER, prediction_info.mask_path), mask)
    cv2.imwrite(os.path.join(PREDICTIONS_OUTPUT_FOLDER, prediction_info.visualization_path), visualization)
    
    return prediction_info


app = Flask(__name__)

DEVICE = "cuda"
MODEL = None


@app.route("/", methods = ["GET","POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files['image']
        tfw_file = request.files['tfw']

        if image_file:
            image_location = os.path.join( 
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)

            if tfw_file:
                tfw_location = os.path.join(
                    UPLOAD_FOLDER,
                    tfw_file.filename
                )
                tfw_file.save(tfw_location)
            else:
                tfw_location = None

            prediction_info = predict(
                image=image_location,
                tfw_path=tfw_location,
                model=MODEL
            )

            return render_template(
                "index.html",
                image_loc=os.path.join(PREDICTIONS_OUTPUT_FOLDER, prediction_info.visualization_path),
                visualization_path=prediction_info.visualization_path,
                mask_path=prediction_info.mask_path,
                shapefile_path=prediction_info.shapefile_path,
                fp16_mode=fp16_mode
            )

    return render_template("index.html",prediction = 0, image_loc=None, fp16_mode=fp16_mode)


@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    return send_file(os.path.join(PREDICTIONS_OUTPUT_FOLDER, filename))


def run_web_ui():
    global MODEL

    create_folders()
    MODEL = init_segmentation_model()
    app.run(port = 8011, debug = True, host="0.0.0.0")


if __name__=="__main__":
    run_web_ui()
