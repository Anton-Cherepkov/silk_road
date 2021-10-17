import os
import cv2
import numpy
from xml.dom import minidom
import cv2  
from pathlib import Path


def get_list_based_lines(polylines_list):
    if len(polylines_list)==0:
        return False

    lines = []
    for i in range(len(polylines_list)):
        temp_points = str(polylines_list[i].attributes['points'].value)
        res = temp_points.split(';')
        points = []
        for x in res:
            points.append((int(float(x.split(',')[0])), int(float(x.split(',')[1]))))
        lines.append(points)
    return lines


def draw_lines(lines,width,height,line_thickness,color=(255,255,255)):
    image = np.zeros((height,width,3), np.uint8)
    for line in lines:
        for i,point in enumerate(line[1:]):
            previous_point = line[i]
            cv2.line(image, previous_point, point,color, thickness=line_thickness)
    return image        


def draw_lines_from_polylines(polylines, dst_folder):
    label2polylines = defaultdict(list)
    polylines_roads = [x for x in polylines if x.getAttribute('label')=='road']
    polylines_looks_like = [x for x in polylines if x.getAttribute('label')=='looks_like']

    lines_roads = get_list_based_lines(polylines_roads)
    lines_looks_like = get_list_based_lines(polylines_looks_like)
    
    img = draw_lines(lines_roads,width,height,line_thickness)
    cv2.imwrite(dst_folder / 'roads/id_{}.png'.format(img_id),img)
        
    img = draw_lines(lines_looks_like,width,height,line_thickness)
    cv2.imwrite(dst_folder / 'looks_like/id_{}.png'.format(img_id),img) 
        


width = 5000
height = 5000
line_thickness = 10
path_to_xml = 'data/data/annotations.xml'
dst_folder = Path("data/data/non_cropped/masks")


xmldoc = minidom.parse(path_to_xml)
images = xmldoc.getElementsByTagName("image")

for image in images:
    img_id = image.getAttribute('id')
    polylines = image.getElementsByTagName("polyline")
    draw_lines_from_polylines(polylines, dst_folder)
    

      


