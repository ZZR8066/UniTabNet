import re
import copy
import Polygon
import numpy as np
from bs4 import BeautifulSoup as bs


def extend_text_lines(cells, lines, text_contetns):
    def segmentation_to_polygon(segmentation):
        polygon = Polygon.Polygon()
        for contour in segmentation:
            polygon = polygon + Polygon.Polygon(contour)
        return polygon

    lines = copy.deepcopy(lines)

    cells_poly = [segmentation_to_polygon([item]) for item in cells]
    lines_poly = [segmentation_to_polygon([item]) for item in lines]

    assign_ids = dict()
    for idx in range(len(cells_poly)):
        assign_ids[idx] = list()

    for line_idx, line_poly in enumerate(lines_poly):
        if line_poly.area() == 0:
            continue
        line_area = line_poly.area()
        max_overlap = 0
        max_overlap_idx = None
        for cell_idx, cell_poly in enumerate(cells_poly):
            overlap = (cell_poly & line_poly).area() / line_area
            if overlap > max_overlap:
                max_overlap_idx = cell_idx
                max_overlap = overlap
            if overlap > 0.99:
                break
        if max_overlap > 0:
            assign_ids[max_overlap_idx].append(line_idx)
    
    cell_transcripts = ['None'] * len(cells)
    for idx, value in assign_ids.items():
        sorted(value)
        value = ''.join([text_contetns[item].replace('<','').replace('>','') for item in value])
        cell_transcripts[idx] = value
        
    return cell_transcripts


def table_to_html(structures, row_spans, col_spans, cell_polys, line_polys, text_contetns):

    cell_transcripts = extend_text_lines(cell_polys, line_polys, text_contetns)

    cells = list()
    tokens = ['<thead>', '</thead>', '<tbody>', '<tr>']
    for idx in range(len(structures)):
        if structures[idx] == '<NL>':
            tokens.append('</tr>')
            tokens.append('<tr>')
        elif structures[idx] == '<C>':
            if row_spans[idx] == 1 and col_spans[idx] == 1:
                tokens.append('<td>')
                tokens.append('</td>')
            else:
                tokens.append('<td')
                if row_spans[idx] > 1:
                    tokens.append(' rowspan="%d"' % row_spans[idx])
                if col_spans[idx] > 1:
                    tokens.append(' colspan="%d"' % col_spans[idx])
                tokens.append('>')
                tokens.append('</td>')

            cell = dict()
            cell['tokens'] = cell_transcripts[idx]
            cell['bbox'] = cell_polys[idx]
            cell['rowspan'] = row_spans[idx]
            cell['colspan'] = col_spans[idx]
            cells.append(cell)
    
    if tokens[-1] == '<tr>':
        tokens = tokens[:-1]
    tokens.append('</tbody>')

    html = dict(
        html=dict(
            cells=cells,
            structure=dict(
                tokens=tokens
            )
        )
    )

    return html
    

def format_html(html):
    html_string = '''<html><body><table>%s</table></body></html>''' % ''.join(html['html']['structure']['tokens'])
    cell_nodes = list(re.finditer(r'(<td[^<>]*>)(</td>)', html_string))
    assert len(cell_nodes) == len(html['html']['cells']), 'Number of cells defined in tags does not match the length of cells'
    cells = [''.join(c['tokens']) for c in html['html']['cells']]
    offset = 0
    for n, cell in zip(cell_nodes, cells):
        html_string = html_string[:n.end(1) + offset] + cell + html_string[n.start(2) + offset:]
        offset += len(cell)
    return html_string