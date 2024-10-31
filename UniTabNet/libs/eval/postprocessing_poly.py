import statistics
import numpy as np
import Polygon
import copy


def polynms_cpu(polys, scores, iou_threshold, score_threshold=0.0):
    """CPU Polygon NMS implementations.

    The input must be numpy array. 

    Arguments:
        polys (np.ndarray): polys in shape (N, 8). format is (x1, y1, x2, y2, x3, y3, x4, y4)
        scores (np.ndarray): scores in shape (N, ).
        iou_threshold (float): IoU threshold for NMS.
        score_threshold (float): score threshold for NMS.
        max_num (int): maximum number of boxes after NMS.

    Returns:
        dets (numpy array): kept dets(boxes and scores) 
        keep_inds (numpy array): kept indice
    """
    x1 = np.min(polys[:, 0::2], axis=1)
    y1 = np.min(polys[:, 1::2], axis=1)
    x2 = np.max(polys[:, 0::2], axis=1)
    y2 = np.max(polys[:, 1::2], axis=1)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polygons = [Polygon.Polygon(poly.reshape(-1,2).tolist()) for poly in polys]
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)

        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = (polygons[i] & polygons[tmp_order[j]]).area() / (polygons[i] + polygons[tmp_order[j]]).area()
            hbb_ovr[h_inds[j]] = iou

        inds = np.where(hbb_ovr <= iou_threshold)[0]
        order = order[inds + 1]

    polys_keep = polys[np.array(keep)]
    scores_keep = scores[np.array(keep)]

    polys_keep = polys_keep[np.where(scores_keep >= score_threshold)[0]]
    keep_inds = np.array(keep)[np.where(scores_keep >= score_threshold)[0]]
    scores_keep = scores_keep[np.where(scores_keep >= score_threshold)[0]]

    dets = np.concatenate((polys_keep, scores_keep[:,None]), axis=-1)
    
    return dets, keep_inds


def poly_nms_without_score(polys, iou_threshold=0.5, score_threshold=0.5):
    '''
    Apply the nms on the output result from the model

    Arguments:
        result: [class_idx][bbox_idx] the format is [x1, y1, x2, y2, x3, y3, x4, y4, scores]
        iou_threshold (float): IoU threshold for NMS.
        score_threshold (float): score threshold for keep avail bbox.

    Returns:
        dets: boxes with scores
    '''
    polys = np.array(polys).reshape(-1, 8)
    scores = np.ones_like(polys[:,0])
    dets, inds = polynms_cpu(polys=polys, scores=scores, iou_threshold=iou_threshold, score_threshold=score_threshold)
    
    return dets[:, :-1]


def result2tablecell(result, iou_threshold=0.5, score_threshold=0.5):
    '''
    Apply the nms on the output result from the model

    Arguments:
        result: [class_idx][bbox_idx] the format is [x1, y1, x2, y2, x3, y3, x4, y4, scores]
        iou_threshold (float): IoU threshold for NMS.
        score_threshold (float): score threshold for keep avail bbox.

    Returns:
        dets: boxes with scores
    '''
    polys = np.ascontiguousarray(result[0][:,:8])
    scores = np.ascontiguousarray(result[0][:,-1])
    dets, inds = polynms_cpu(polys=polys, scores=scores, iou_threshold=iou_threshold, score_threshold=score_threshold)
    
    if len(dets) == 0:
        dets = np.array([[0,0,0,0,0,0,0,0,0]]).astype(np.float32)
        
    return dets


def group_cell_pts(cells, r=0.3, dis_threshold=30):
    '''
    group the surround pts of cells to the points

    Arguments:
        cells: polys with scores
        r: radius for group points
        dis_threshold: distance threshold for group points

    Returns:
        gcells: grouped boxes with scores
    '''
    grouped_cells = list()
    for cell in cells:
        x1, y1, x2, y2, x3, y3, x4, y4 = [float(item) for item in cell]
        w = max(abs(x2-x1), abs(x3-x4))
        h = max(abs(y4-y1), abs(y3-y2))
        grouped_cells.append(dict(vertexs=[[x1,y1,False],[x2,y2,False],\
            [x3,y3,False],[x4,y4,False]],w=w, h=h))

    for i, cell_i in enumerate(grouped_cells):
        vertexs_i = cell_i['vertexs']
        w_i = cell_i['w']
        h_i = cell_i['h']
        xoffset1 = max(w_i*r, 4.0)
        yoffset1 = max(h_i*r, 4.0)
        for l, vertex_il in enumerate(vertexs_i):
            x_il, y_il, state_il = vertex_il
            if not state_il: # has not been grouped
                keep_inds = []
                keep_pts = []
                for j, cell_j in enumerate(grouped_cells):
                    if j == i:
                        continue
                    vertexs_j = cell_j['vertexs']
                    w_j = cell_j['w']
                    h_j = cell_j['h']
                    xoffset2 = max(w_j*r, 4.0)
                    yoffset2 = max(h_j*r, 4.0)
                    for k, vertex_jk in enumerate(vertexs_j):
                        x_jk, y_jk, state_jk = vertex_jk
                        if not state_jk: # has not been grouped
                            xdist = abs(x_il - x_jk)
                            ydist = abs(y_il - y_jk)
                            vector1 = np.array([x_il, y_il])
                            vector2 = np.array([x_jk, y_jk])
                            dist = np.sqrt(np.sum(np.square(vector1-vector2)))
                            if xdist > xoffset1 or xdist > xoffset2 or ydist > yoffset1 or \
                                ydist > yoffset2 or dist > dis_threshold:
                                continue
                            else:
                                keep_inds.append([j,k])
                                keep_pts.append([x_jk, y_jk])
            
                # average the (x_il, y_il) surrounding pts
                keep_inds.append([i,l])
                keep_pts.append([x_il,y_il])
                ptx, pty = np.array(keep_pts).mean(0)
                # set the state and pts 
                for ind in keep_inds:
                    cell_idx, vertex_idx = ind
                    grouped_cells[cell_idx]['vertexs'][vertex_idx][0] = int(ptx)
                    grouped_cells[cell_idx]['vertexs'][vertex_idx][1] = int(pty)
                    grouped_cells[cell_idx]['vertexs'][vertex_idx][-1] = True
    
    # trans dict to np.array
    cell_bboxes = list()
    for cell in grouped_cells:
        vertexs = [vertex[:2] for vertex in cell['vertexs']] # remove state
        vertexs = np.array(vertexs).reshape(-1) # points
        vertexs = [int(vertex) for vertex in vertexs]
        cell_bboxes.append(vertexs)
    
    return cell_bboxes


def sub_merge_celledges(in_celledges, sort_dim):
    '''
    single merge the cell edeges

    Arguments:
        subedges: a set of vertexs of subedges
        sort_dim: according to the x/y (0/1) to sort the points of edge

    Returns:
        tableedges: a set of table edeges
    '''
    celledges = copy.deepcopy(in_celledges)
    table_edges = [celledges[0]]
    for celledge in celledges[1:]:
        is_existed = False # indicate current edge is in table edges list or not
        for table_edge in table_edges:
            if table_edge == celledge:
                continue
            for vertex in celledge:
                if vertex in table_edge:
                    table_edge.extend(celledge)
                    is_existed = True
                    break
            if is_existed:
                break
        if not is_existed:
            table_edges.append(celledge)
    
    new_table_edges = []
    for table_edge in table_edges:
        table_edge = list(set(table_edge))
        sorted_idx = sorted(list(range(len(table_edge))), key=lambda idx: table_edge[idx][sort_dim])
        table_edge = [table_edge[idx] for idx in sorted_idx]
        new_table_edges.append(table_edge)
    return new_table_edges


def merge_celledges(celledges, sort_dim):
    '''
    merge the cell edeges

    Arguments:
        subedges: a set of vertexs of subedges
        sort_dim: according to the x/y (0/1) to sort the points of edge

    Returns:
        tableedges: a set of table edeges
    '''
    pre_table_edges = celledges
    while True:
        cur_table_edges = sub_merge_celledges(pre_table_edges, sort_dim)
        if len(cur_table_edges) == len(pre_table_edges):
            break
        else:
            pre_table_edges = cur_table_edges

    mean_values = [statistics.mean([pt[1-sort_dim] for pt in edge]) for edge in cur_table_edges]
    sorted_idx = sorted(list(range(len(cur_table_edges))), key=lambda idx: mean_values[idx])
    cur_table_edges = [cur_table_edges[idx] for idx in sorted_idx]
    return cur_table_edges


def cells2logicalstruct(cells):
    '''
    parse the cells to logical structure
    Fisrt merger table line
    Then apply the index of the merged table lines, which is
    the logical structurte of the table

    Arguments:
        cells: grouped cell boxes with scores
        verticaledges: the merged vertical edges
        horizontaledges: the merged horizontal edges

    Returns:
        ls: logical structure of each cell [[xs,xe,ys,ye]]
    '''
    cellleftedges = []
    cellrightedges = []
    cellupedges = []
    celldownedges = []
    for cell in cells:
        vertexs = [(int(cell[2*idx]), int(cell[2*idx+1])) for idx in range(4)]
        cellleftedges.append([vertexs[0], vertexs[3]])
        cellrightedges.append([vertexs[1], vertexs[2]])
        cellupedges.append([vertexs[0], vertexs[1]])
        celldownedges.append([vertexs[2], vertexs[3]])
    
    verticaledges = []
    verticaledges = cellleftedges
    [verticaledges.append(edge) for edge in cellrightedges if edge not in verticaledges]
    verticaledges = merge_celledges(verticaledges, 1)

    horizontaledges = []
    horizontaledges = cellupedges
    [horizontaledges.append(edge) for edge in celldownedges if edge not in horizontaledges]
    horizontaledges = merge_celledges(horizontaledges, 0)

    cells_info = []
    for cell in cells:
        vertexs = [(int(cell[2*idx]), int(cell[2*idx+1])) for idx in range(4)]

        column_start_idx = -1
        column_end_idx = -1
        for index, edge in enumerate(verticaledges):
            if vertexs[0] in edge: # upleft point
                column_start_idx = index
            if vertexs[2] in edge: # downright point
                column_end_idx = index
            if column_start_idx != -1 and column_end_idx != -1:
                break

        row_start_idx = -1
        row_end_idx = -1
        for index, edge in enumerate(horizontaledges):
            if vertexs[0] in edge: # upleft point
                row_start_idx = index
            if vertexs[2] in edge: # downright point
                row_end_idx = index
            if row_start_idx != -1 and row_end_idx != -1:
                break
        
        # correct assert error
        if row_start_idx >= row_end_idx or min(row_start_idx, row_end_idx) == -1:
            if row_start_idx != -1:
                row_end_idx = row_start_idx + 1
            elif row_end_idx != -1:
                row_start_idx = max(0, row_end_idx - 1)
            else:
                row_start_idx = 0
                row_end_idx = 1
                
        if column_start_idx >= column_end_idx or min(column_start_idx, column_end_idx) == -1:
            if column_start_idx != -1:
                column_end_idx = column_start_idx + 1
            elif column_end_idx != -1:
                column_start_idx = max(0, column_end_idx - 1)
            else:
                column_start_idx = 0
                column_end_idx =1

        cells_info.append(dict(poly=vertexs, row_start_idx=row_start_idx, row_end_idx=row_end_idx-1, \
            column_start_idx=column_start_idx, column_end_idx=column_end_idx-1))

    return cells_info


def parse_logicalstruct(result, iou_threshold=0.5, score_threshold=0.7, r=0.5, dis_threshold=30):
    '''
    Post processing for parsing the result to logical structure

    Arguments:
        result:[class_idx][bbox_idx].
        iou_threshold (float): IoU threshold for NMS.
        score_threshold (float): score threshold for NMS.
        r: radius for group points
        dis_threshold: distance threshold for group points
        

    Returns:
        cell_infos: logical/physical structure [dict(poly=vertexs, row_start_idx=row_start_idx, row_end_idx=row_end_idx, 
            column_start_idx=column_start_idx, column_end_idx=column_end_idx)]
    '''
    table_cells_nms = result2tablecell(result, iou_threshold=iou_threshold, score_threshold=score_threshold)
    table_cells_group = group_cell_pts(table_cells_nms, r=r, dis_threshold=dis_threshold)
    cell_infos = cells2logicalstruct(table_cells_group)
    return table_cells_nms, table_cells_group, cell_infos