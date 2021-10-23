import os
import cv2
import numpy as np
from xml.dom import minidom
import cv2  
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def get_list_based_lines(polylines_list):
    lines = []
    for i in range(len(polylines_list)):
        temp_points = str(polylines_list[i].attributes['points'].value)
        res = temp_points.split(';')
        points = []
        for x in res:
            points.append((int(float(x.split(',')[0])), int(float(x.split(',')[1]))))
        lines.append(points)
    return lines


def draw_lines(lines,width,height,line_thickness,color=(1,)):
    image = np.zeros((height,width), np.uint8)
    for line in lines:
        for i,point in enumerate(line[1:]):
            previous_point = line[i]
            cv2.line(image, previous_point, point,color, thickness=line_thickness)
    return image        


def draw_lines_from_polylines(name, polylines, dst_folder):
    label2curves = {
        "looks_like": [],
        "road": []
    }

    for curves in polylines:
        label2curves[curves.getAttribute("label")].append(curves)
    
    for label, curves in label2curves.items():
        dst = dst_folder / f"{label}/{name}.png"
        dst.parent.mkdir(exist_ok=True, parents=True)

        img = draw_lines(get_list_based_lines(curves), width, height, line_thickness)
        cv2.imwrite(str(dst), img)
        

width = 5000
height = 5000
line_thickness = 10
path_to_xml = 'data/data/annotations.xml'
dst_folder = Path("data/data/non_cropped/masks")


xmldoc = minidom.parse(path_to_xml)
images = xmldoc.getElementsByTagName("image")

for image in tqdm(images):
    name = Path(image.getAttribute('name')).stem
    polylines = image.getElementsByTagName("polyline")
    draw_lines_from_polylines(name, polylines, dst_folder)
    

      


