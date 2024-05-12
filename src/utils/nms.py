import numpy as np

def nms(bboxes, scores, thres=0.5):
    
    sorted_indexes = np.argsort(scores)[::-1]
    
    x_min = bboxes[:,0]
    y_min = bboxes[:,1]
    x_max = bboxes[:,2]
    y_max = bboxes[:,3]
    
    areas = abs((x_max - x_min + 1) * (y_max - y_min + 1))
    
    keep = []
    
    while sorted_indexes.size:
        curr_idx = sorted_indexes[0]
        keep.append(curr_idx)
        
        overlap_xmins = np.maximum(x_min[curr_idx], x_min)
        overlap_ymins = np.maximum(y_min[curr_idx], y_min)
        overlap_xmaxs = np.minimum(x_max[curr_idx], x_max)
        overlap_ymaxs = np.minimum(y_max[curr_idx], y_max)
        
        overlap_widths = np.maximum(0, overlap_xmaxs - overlap_xmins + 1)
        overlap_heights = np.maximum(0, overlap_ymaxs - overlap_ymins + 1)
        overlap_areas = overlap_widths * overlap_heights
        
        ious = overlap_areas / (areas[curr_idx] + areas - overlap_areas)
        
        deleted_idx = np.where(ious[sorted_indexes] > thres)[0]
        sorted_indexes = np.delete(sorted_indexes, deleted_idx)
        
    return bboxes[keep]
        
        
    
if __name__=='__main__':
    # n x 4
    bboxes = [
            [12,311,84,362],
            [10,300,80,360],
            [362,330,500,389],
            [360,330,500,380],
            [175,327,252,364],
            [170,320,250,360],
            [108,325,150,353],
            [100,320,150,350]
        ]
    bboxes = np.array(bboxes)
    scores = np.array([0.68,0.98,0.89,0.88,0.79,0.78,0.69,0.99])
    
    keep = nms(bboxes, scores)
    print(keep)