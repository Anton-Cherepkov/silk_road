import cv2
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes, medial_axis
from skimage.morphology import erosion, dilation, opening, closing, disk
import numpy as np
import networkx as nx
import postprocessing.sknw as sknw


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
        curves.append(
            G[src][dst][0]["pts"][:, ::-1].astype(np.int32)
        )
    
    return curves


def draw_polyline(img, polyline):
    img = img.copy()
    for curve in polyline:
        for i in range(len(curve) - 1):
            cv2.line(img, curve[i], curve[i + 1], (255, 0, 0), thickness=5)
        
    return img


def get_postprocessing_visualization(img, mask):
    if isinstance(img, str):
        img = cv2.imread(img)

    _, _, G = img_to_ske_G(mask)
    curves = nx2polyline(G)
    postpocessing_visualization = draw_polyline(img, curves)
    return postpocessing_visualization
