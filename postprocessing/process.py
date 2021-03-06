from dataclasses import dataclass
import cv2
from numpy.core.fromnumeric import shape
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes, medial_axis
from skimage.morphology import erosion, dilation, opening, closing, disk
import numpy as np
import networkx as nx
import fiona
import postprocessing.sknw as sknw
from postprocessing.tfw import pixel_coord_to_world_coords, TFWCoordinates, read_tfw_file
import shutil
from pathlib import Path
from typing import Optional, List


@dataclass
class PostprocessingResult:
    postpocessing_visualization: np.ndarray
    shapefile_zip_path: Optional[str]
    roads_curves: List[np.ndarray]


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


def preprocess(img, small_size=300, hole_size=300):
    '''
    http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.remove_small_holes
    hole_size in remove_small_objects is the maximum area, in pixels of the
    hole
    '''

    # sometimes get a memory error with this approach
    assert img.size < 10000000000
    
    img = img.astype(np.bool)        
    remove_small_objects(img, small_size, in_place=True)
    remove_small_holes(img, hole_size, in_place=True)

    return img


def make_skeleton(img, small_size=150, hole_size=150, fix_borders=True, replicate=5, clip=2):
    img = cv2.copyMakeBorder(
        img, 
        replicate, 
        replicate, 
        replicate, 
        replicate, 
        cv2.BORDER_REPLICATE
    )  
    
    img = preprocess(img, hole_size=hole_size)
    
    ske = skeletonize(img).astype(np.uint16)
    
    rec = replicate + clip
    
    if fix_borders:
        ske = ske[rec:-rec, rec:-rec]
        ske = cv2.copyMakeBorder(ske, clip, clip, clip, clip, cv2.BORDER_CONSTANT, value=0)

        img = img[replicate:-replicate,replicate:-replicate]

    return img, ske


def flatten(l):
    return [item for sublist in l for item in sublist]


def remove_small_terminal(G, weight='weight', min_weight_val=30, 
                          pix_extent=1300, edge_buffer=4):
    '''Remove small terminals, if a node in the terminal is within edge_buffer
    of the the graph edge, keep it'''
    deg = dict(G.degree())
    terminal_points = [i for i, d in deg.items() if d == 1]

    edges = list(G.edges())
    for s, e in edges:
        if s == e:
            sum_len = 0
            vals = flatten([[v] for v in G[s][s].values()])
            for ix, val in enumerate(vals):
                sum_len += len(val['pts'])
            if sum_len < 3:
                G.remove_edge(s, e)
                continue
            
        # check if at edge
        sx, sy = G.nodes[s]['o']
        ex, ey = G.nodes[e]['o']
        edge_point = False
        for ptmp in [sx, sy, ex, ey]:
            if (ptmp < (0 + edge_buffer)) or (ptmp > (pix_extent - edge_buffer)):
                edge_point = True
            else:
                continue
        # don't remove edges near the edge of the image
        if edge_point:
            continue

        vals = flatten([[v] for v in G[s][e].values()])
        for ix, val in enumerate(vals):
            if s in terminal_points and val.get(weight, 0) < min_weight_val:
                G.remove_node(s)
            if e in terminal_points and val.get(weight, 0) < min_weight_val:
                G.remove_node(e)
    return


def img_to_ske_G(
    img, 
    small_size=150,
    hole_size=150,
    min_spur_length_pix=16
):
    img_refine, ske = make_skeleton(
        img,
        small_size=small_size,
        hole_size=hole_size
    )
    
    assert np.max(ske.shape) <= 32767
    
    G = sknw.build_sknw(ske, multi=True)
    
    for itmp in range(8):
        ntmp0 = len(G.nodes())
        
        # sknw attaches a 'weight' property that is the length in pixels
        pix_extent = np.max(ske.shape)
        remove_small_terminal(G, weight='weight',
                              min_weight_val=min_spur_length_pix,
                              pix_extent=pix_extent)
        # kill the loop if we stopped removing nodes
        ntmp1 = len(G.nodes())
        if ntmp0 == ntmp1:
            break
        else:
            continue
    
    ebunch = nx.selfloop_edges(G)
    G.remove_edges_from(list(ebunch))
    
    return img_refine, ske, G

        
def nx2polyline(G):
    curves = []
    for src, dst in G.edges():
        points = G[src][dst][0]["pts"][:, ::-1].astype(np.int32)

        if len(points) > 1:
            curves.append(
                points
            )
    
    return curves


def draw_polyline(img, polyline):
    img = img.copy()
    for curve in polyline:
        for i in range(len(curve) - 1):
            cv2.line(img, curve[i], curve[i + 1], (255, 0, 0), thickness=5)
        
    return img


def polylines2shapefile(polylines: List[List[np.ndarray]], tfws: List[TFWCoordinates], path: str, crs: str = "EPSG:32637"):
    schema = {
        "geometry": "LineString",
    }
    
    with fiona.open(path, mode='w', 
                    driver='ESRI Shapefile',
                    schema=schema, 
                    crs=crs) as f:
        for polyline, tfw in zip(polylines, tfws):
            for line in polyline:
                points = [pixel_coord_to_world_coords(tfw, x, y) for x, y in line]
                f.write({
                    "geometry": {
                        "type": "LineString",
                        "coordinates": points
                    }
                })


def polyline2shapefile(polyline: np.ndarray, path: str, tfw: TFWCoordinates, crs: str = "EPSG:32637"):
    polylines2shapefile([polyline], [tfw], path, crs)
            
    
def zip_and_remove(folder):
    if folder.endswith("/"):
        folder = folder[:-1]
    shutil.make_archive(folder, 'zip', folder)
    shutil.rmtree(folder)
    
    return folder + ".zip"

  
def do_postprocessing(img, mask, tfw_path: Optional[str], shapefile_folder_path: str) -> PostprocessingResult:
    if isinstance(img, str):
        img = cv2.imread(img)

    _, _, G = img_to_ske_G(mask)
    curves = nx2polyline(G)
    postpocessing_visualization = draw_polyline(img, curves)

    if tfw_path is not None:
        tfw = read_tfw_file(tfw_path)
        polyline2shapefile(curves, shapefile_folder_path, tfw)
        shapefile_zip_path = zip_and_remove(shapefile_folder_path)
    else:
        shapefile_zip_path = None

    result = PostprocessingResult(
        postpocessing_visualization=postpocessing_visualization,
        shapefile_zip_path=shapefile_zip_path,
        roads_curves=curves
    )

    return result


if __name__ == "__main__":
    img = cv2.imread("3560-825.tif")
    mask = np.load("3560-825.npy")
    mask_postprocessed, ske, G = img_to_ske_G(mask)
    
    roads_polyline = nx2polyline(G)
    
    cv2.imwrite("mask_postprocessed.png", draw_mask(img, mask_postprocessed))
    cv2.imwrite("ske.png", draw_mask(img, ske))
    cv2.imwrite("graph.png", draw_polyline(img, roads_polyline))
    
    tfw = read_tfw_file("3560-825.tfw")
    polyline2shapefile(roads_polyline, "3560-825", tfw)
    
    input()
    
    zip_and_remove("3560-825")

