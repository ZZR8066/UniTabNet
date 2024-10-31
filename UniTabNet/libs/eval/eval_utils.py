import Polygon
import numpy as np


def assign_content(gt_cells, det_cells, iou_threshold=0.6):
    '''assigned consistent contents according the IOU between gt and detection result
    
    Arguments:
        gt_cells: [dict(startcol=startcol,startrow=startrow,endcol=endcol,endrow=endrow,poly=poly,content='')]
        det_cells: [dict(startcol=startcol,startrow=startrow,endcol=endcol,endrow=endrow,poly=poly,content='')]

    Returns:
        gt_cells: [dict(startcol=startcol,startrow=startrow,endcol=endcol,endrow=endrow,poly=poly,content=content)]
        det_cells: [dict(startcol=startcol,startrow=startrow,endcol=endcol,endrow=endrow,poly=poly,content=content)]
    '''

    gt_polygons = [Polygon.Polygon(item['poly']) for item in gt_cells]
    det_polygons = [Polygon.Polygon(item['poly']) for item in det_cells]

    for gt_idx, gt_polygon in enumerate(gt_polygons):
        if gt_polygon.area() == 0:
            continue
        gt_area = gt_polygon.area()
        max_overlap = 0
        max_overlap_idx = None
        for det_idx, det_polygon in enumerate(det_polygons):
            overlap = (gt_polygon & det_polygon).area() / (gt_polygon + det_polygon).area( )
            if overlap > max_overlap:
                max_overlap_idx = det_idx
                max_overlap = overlap
        if max_overlap > iou_threshold:
            det_cells[max_overlap_idx]['content'] = gt_cells[gt_idx]['content']


def trans2evaltype(label, predict, iou_threshold=0.0):
    '''translate the prediction and ground truth to eval type

    Arguments:
        label: the path of label which is annotated as [x1,y1,x2,y2,x3,y3,x4,y4,startcol,startrow,endcol,endrow]
        predict: post-processing results which is define as [dict(poly=vertexs, row_start_idx=row_start_idx, row_end_idx=row_end_idx, 
            column_start_idx=column_start_idx, column_end_idx=column_end_idx)]
        iou_threshold (float): the min threshold for aligning det_box to gt_box

    Returns:
        gt_cells: a dict for ground truth cells
        det_cells: a dict for model predict cells
    '''
    gt_cells = []
    for idx, cell in enumerate(label):
        poly = np.array(cell['poly']).reshape(-1,2).tolist()
        startrow = cell['row_start_idx']
        endrow = cell['row_end_idx']
        startcol = cell['column_start_idx']
        endcol = cell['column_end_idx']
        gt_cells.append(dict(
            start_col=startcol,
            start_row=startrow,
            end_col=endcol,
            end_row=endrow,
            poly=poly,
            content=[str(idx)]
        ))
        assert endcol >= startcol
        assert endrow >= startrow

    det_cells = []
    for cell in predict:
        poly = np.array(cell['poly']).reshape(-1,2).tolist()
        startrow = cell['row_start_idx']
        endrow = cell['row_end_idx']
        startcol = cell['column_start_idx']
        endcol = cell['column_end_idx']
        det_cells.append(dict(
            start_col=startcol,
            start_row=startrow,
            end_col=endcol,
            end_row=endrow,
            poly=poly,
            content=''
        ))
        assert endcol >= startcol
        assert endrow >= startrow
    
    assign_content(gt_cells, det_cells, iou_threshold)
    return dict(cells=gt_cells), dict(cells=det_cells)