import os
import cv2
import sys
import tqdm
import json
import glob
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
from PIL import Image
import numpy as np
from libs.eval.format_translate import table_to_html, format_html
from libs.eval.metric import TEDSMetric

html_dirs = [
    # '/train21/mmu/permanent/zrzhang6/TableQA/code/unitable-v13/experiments/table_structure/checkpoint-3865',
    "/train21/mmu/permanent/zrzhang6/TableQA/code/unitable-v13/experiments/table_structure/checkpoint-7730",
    "/train21/mmu/permanent/zrzhang6/TableQA/code/unitable-v13/experiments/table_structure/checkpoint-11595",
    "/train21/mmu/permanent/zrzhang6/TableQA/code/unitable-v13/experiments/table_structure/checkpoint-15460",
    "/train21/mmu/permanent/zrzhang6/TableQA/code/unitable-v13/experiments/table_structure/checkpoint-19325",
]

for html_dir in html_dirs:
    pred_htmls = []
    label_htmls = []

    wired_html_dir = os.path.join(html_dir, 'wired_htmls')
    for idx in range(int(1e8)):
        pred_html_path = os.path.join(wired_html_dir, '%04d_pred_html.json' % idx)
        label_html_path = os.path.join(wired_html_dir, '%04d_label_html.json' % idx)
        if not os.path.exists(pred_html_path) or not os.path.exists(label_html_path):
            break
        pred_htmls.append(format_html(json.load(open(pred_html_path, 'r'))))
        label_htmls.append(format_html(json.load(open(label_html_path, 'r'))))

    wireless_html_dir = os.path.join(html_dir, 'wireless_htmls')
    for idx in range(int(1e8)):
        pred_html_path = os.path.join(wireless_html_dir, '%04d_pred_html.json' % idx)
        label_html_path = os.path.join(wireless_html_dir, '%04d_label_html.json' % idx)
        if not os.path.exists(pred_html_path) or not os.path.exists(label_html_path):
            break
        pred_htmls.append(format_html(json.load(open(pred_html_path, 'r'))))
        label_htmls.append(format_html(json.load(open(label_html_path, 'r'))))

    teds_metric = TEDSMetric(num_workers=30, structure_only=False)
    teds_info = teds_metric(pred_htmls, label_htmls)
    result = 'Total Html files: %d \n' % len(teds_info)
    result += 'TEDS-Struct: %.2f' % (100*sum(teds_info) / len(teds_info))
    with open(os.path.join(html_dir, 'result.txt'), 'w') as f:
        f.write(result)
    print('While evaluate %s\n%s' % (html_dir, result))