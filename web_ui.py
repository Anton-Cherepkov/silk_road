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

from typing import Optional, List, Tuple

from argparse import ArgumentParser
import torch
import glob
import os
from pathlib import Path
from tqdm.auto import tqdm
from dataclasses import asdict, dataclass
from itertools import chain
# import sns
import mmcv
from mmcv.runner import wrap_fp16_model
import cv2
import numpy as np
from datetime import datetime

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

from postprocessing.process import do_postprocessing, PostprocessingResult, polylines2shapefile, read_tfw_file, zip_and_remove


FP_16_MODE = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PREDICTIONS_OUTPUT_FOLDER = "static/predictions"
UPLOAD_FOLDER = 'static/uploads'


@dataclass
class PredictionInformation:
    visualization_path: str
    mask_path: str
    shapefile_path: Optional[str]
    postprocessing_visualization_path: Optional[str]


def get_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--fp16', type=bool, default=False)
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
    cfg.model.test_cfg['stride'] = (512, 512)
    cfg.model.test_cfg['crop_size'] = (512, 512)
    model = init_segmentor(cfg, args.checkpoint, device=DEVICE)
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
    postprocessing_visualization_folder = os.path.join(PREDICTIONS_OUTPUT_FOLDER, "postprocessing_visualization")
    shapefiles_folder = os.path.join(PREDICTIONS_OUTPUT_FOLDER, "shapefiles")
    common_shapefiles_folder = os.path.join(PREDICTIONS_OUTPUT_FOLDER, "common_shapefiles")
    for folder in [masks_folder, visualizations_folder, postprocessing_visualization_folder, UPLOAD_FOLDER, shapefiles_folder, common_shapefiles_folder]:
        os.makedirs(folder, exist_ok=True)


def predict(image, model, tfw_path: Optional[str]) -> Tuple[PredictionInformation, PostprocessingResult]:
    image_name = Path(image).stem

    mask = inference_segmentor(model, image)[0]
    visualization = draw_mask(image, mask)
    postprocessing_result = do_postprocessing(
        image, mask,
        tfw_path=tfw_path, shapefile_folder_path=os.path.join(
            PREDICTIONS_OUTPUT_FOLDER, "shapefiles", image_name)
    )

    prediction_info = PredictionInformation(
        visualization_path=os.path.join("visualization", f"{image_name}.jpg"),
        mask_path=os.path.join("masks", f"{image_name}.npy"),
        postprocessing_visualization_path=os.path.join("postprocessing_visualization", f"{image_name}.jpg"),
        shapefile_path=os.path.relpath(postprocessing_result.shapefile_zip_path, PREDICTIONS_OUTPUT_FOLDER) if postprocessing_result.shapefile_zip_path else None,
    )

    np.save(os.path.join(PREDICTIONS_OUTPUT_FOLDER, prediction_info.mask_path), mask)
    cv2.imwrite(os.path.join(PREDICTIONS_OUTPUT_FOLDER, prediction_info.visualization_path), visualization)
    cv2.imwrite(
        os.path.join(PREDICTIONS_OUTPUT_FOLDER, prediction_info.postprocessing_visualization_path),
        postprocessing_result.postpocessing_visualization,
    )

    return prediction_info, postprocessing_result


def predict_multiple(images: List[str], tfw_paths: List[Optional[str]], model) -> Tuple[List[PredictionInformation], Optional[str]]:
    results = []

    for image, tfw_path in zip(images, tfw_paths):
        results.append(predict(image, model, tfw_path))

    prediction_infos, postprocessing_results = zip(*results)

    if any(tfw_paths):
        polylines, tfws = zip(*[
            (postprocessing_result.roads_curves, read_tfw_file(tfw_path))
            for postprocessing_result, tfw_path in zip(postprocessing_results, tfw_paths)
            if tfw_path is not None
        ])
        all_roads_shapefile = os.path.join(
            PREDICTIONS_OUTPUT_FOLDER,
            "common_shapefiles",
            datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        )
        polylines2shapefile(polylines, tfws, all_roads_shapefile)
        all_roads_shapefile = zip_and_remove(all_roads_shapefile)
        all_roads_shapefile = os.path.relpath(
            all_roads_shapefile,
            PREDICTIONS_OUTPUT_FOLDER
        )
    else:
        all_roads_shapefile = None 

    return prediction_infos, all_roads_shapefile


def make_mapping_filename_to_file(files):
    mapping = dict()
    for file in files:
        mapping[Path(file.filename).stem] = file
    return mapping


def save_corresponding_images_and_tfws(images, tfws):
    names_to_images = make_mapping_filename_to_file(images)
    names_to_tfws = make_mapping_filename_to_file(tfws)

    saved_images = []
    saved_tfws = []

    for name, image in names_to_images.items():
        image_loc = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_loc)

        tfw = names_to_tfws.get(name)
        if tfw:
            tfw_loc = os.path.join(UPLOAD_FOLDER, tfw.filename)
            tfw.save(tfw_loc)
        else:
            tfw_loc = None
        
        saved_images.append(image_loc)
        saved_tfws.append(tfw_loc)

    names = [
        Path(path).stem
        for path in saved_images
    ]

    return names, saved_images, saved_tfws


app = Flask(__name__)

MODEL = None


@app.route("/", methods = ["GET","POST"])
def upload_predict():
    if request.method == "POST":
        images = request.files.getlist("image[]")
        tfws = request.files.getlist("tfw[]")

        names, saved_images, saved_tfws = save_corresponding_images_and_tfws(
            images=images,
            tfws=tfws,
        )

        prediction_infos, all_roads_shapefile = predict_multiple(
            images=saved_images,
            tfw_paths=saved_tfws,
            model=MODEL
        )

        prediction_infos = list(map(asdict, prediction_infos))
        for name, prediction_info in zip(names, prediction_infos):
            prediction_info["name"] = name
        
        print(all_roads_shapefile)

        return render_template(
            "index.html",
            all_roads_shapefile=all_roads_shapefile,
            predictions=prediction_infos,
            fp16_mode=fp16_mode,
            device=DEVICE,
        )

    return render_template(
        "index.html",
        predictions=[],
        fp16_mode=fp16_mode,
        device=DEVICE,
    )


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
