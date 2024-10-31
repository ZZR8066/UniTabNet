import os
import cv2
import sys
import tqdm
import json
import glob
sys.path.append('./')
sys.path.append('../')
from PIL import Image
import numpy as np
from utils import str2image
from libs.eval.format_translate import format_relation
from libs.eval.cal_f1 import evaluate_f1
from bs4 import BeautifulSoup


html_dirs = [
    # "/train21/mmu/permanent/zrzhang6/TableQA/code/unitable-v26/experiments/table_structure/checkpoint-1874",
    # "/train21/mmu/permanent/zrzhang6/TableQA/code/unitable-v26/experiments/table_structure/checkpoint-3748",
    # "/train21/mmu/permanent/zrzhang6/TableQA/code/unitable-v26/experiments/table_structure/checkpoint-5622",
    # "/train21/mmu/permanent/zrzhang6/TableQA/code/unitable-v26/experiments/table_structure/checkpoint-7496",
    # "/train21/mmu/permanent/zrzhang6/TableQA/code/unitable-v26/experiments/table_structure/checkpoint-9370",
    # "/train21/mmu/permanent/zrzhang6/TableQA/code/unitable-v26/experiments/table_structure/checkpoint-11244",
    # "/train21/mmu/permanent/zrzhang6/TableQA/code/unitable-v26/experiments/table_structure/checkpoint-13118",
    # "/train21/mmu/permanent/zrzhang6/TableQA/code/unitable-v26/experiments/table_structure/checkpoint-14992",
    # "/train21/mmu/permanent/zrzhang6/TableQA/code/unitable-v26/experiments/table_structure/checkpoint-16866",
    # "/train21/mmu/permanent/zrzhang6/TableQA/code/unitable-v26/experiments/table_structure/checkpoint-18740",
    # "/train21/mmu/permanent/zrzhang6/TableQA/code/unitable-v26/experiments/table_structure/checkpoint-20614",
    # "/train21/mmu/permanent/zrzhang6/TableQA/code/unitable-v26/experiments/table_structure/checkpoint-22488",
    # "/train21/mmu/permanent/zrzhang6/TableQA/code/unitable-v26/experiments/table_structure/checkpoint-24362",
    # "/train21/mmu/permanent/zrzhang6/TableQA/code/unitable-v26/experiments/table_structure/checkpoint-26236",
    # "/train21/mmu/permanent/zrzhang6/TableQA/code/unitable-v26/experiments/table_structure/checkpoint-28110",
    # "/train21/mmu/permanent/zrzhang6/TableQA/code/unitable-v26/experiments/table_structure/checkpoint-29984",
    # "/train21/mmu/permanent/zrzhang6/TableQA/code/unitable-v26/experiments/table_structure/checkpoint-31858",
    "/train21/mmu/permanent/zrzhang6/TableQA/code/unitable-v26/experiments/table_structure/checkpoint-33732",

    # "/train21/mmu/permanent/zrzhang6/TableQA/code/unitable-v26/experiments/table_structure_ml/checkpoint-33732",
]


def remove_span_attributes_with_bs(html):
    soup = BeautifulSoup(html, 'html.parser')
    # 查找所有td标签
    for td in soup.find_all('td'):
        # 移除rowspan和colspan属性
        if 'rowspan' in td.attrs:
            del td['rowspan']
        if 'colspan' in td.attrs:
            del td['colspan']
    return str(soup)


def format_relation_mp(htmls, num_workers):
    def _worker(idxs, htmls, result_queue):
        sub_info = dict()
        for idx in tqdm.tqdm(idxs):
            sub_info[idx] = format_relation(json.load(open(htmls[idx], 'r')))
        result_queue.put(sub_info)
    
    import multiprocessing
    num_workers = 30
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    idxs = list(range(len(htmls)))
    workers = list()
    for worker_idx in range(num_workers):
        worker = multiprocessing.Process(
            target=_worker,
            args=(
                idxs[worker_idx::num_workers],
                htmls,
                result_queue
            )
        )
        worker.daemon = True
        worker.start()
        workers.append(worker)

    info_total = dict()
    for _ in range(num_workers):
       sub_info = result_queue.get()
       info_total.update(sub_info)
    
    infos = [info_total[idx] for idx in range(len(info_total.keys()))]
    return infos


for html_dir in html_dirs:
    wired_html_dir = os.path.join(html_dir, 'wired_htmls')
    pred_wired_htmls = glob.glob(os.path.join(wired_html_dir, "*_pred_html.json"))
    label_wired_htmls = glob.glob(os.path.join(wired_html_dir, "*_label_html.json"))
    pred_wired_htmls = sorted(pred_wired_htmls)
    label_wired_htmls = sorted(label_wired_htmls)
    assert len(pred_wired_htmls) == len(label_wired_htmls)

    pred_wired_relations = format_relation_mp(pred_wired_htmls, 30)
    label_wired_relations = format_relation_mp(label_wired_htmls, 30)

    wireless_html_dir = os.path.join(html_dir, 'wireless_htmls')
    pred_wireless_htmls = glob.glob(os.path.join(wireless_html_dir, "*_pred_html.json"))
    label_wireless_htmls = glob.glob(os.path.join(wireless_html_dir, "*_label_html.json"))
    pred_wireless_htmls = sorted(pred_wireless_htmls)
    label_wireless_htmls = sorted(label_wireless_htmls)
    assert len(pred_wireless_htmls) == len(label_wireless_htmls)

    pred_wireless_relations = format_relation_mp(pred_wireless_htmls, 30)
    label_wireless_relations = format_relation_mp(label_wireless_htmls, 30)

    result = "----------------------F1-Measure-------------------------------\n"
    wired_f1 = evaluate_f1(label_wired_relations, pred_wired_relations, num_workers=0)
    wired_p = [item[0] for item in wired_f1]
    wired_r = [item[1] for item in wired_f1]
    p = sum(wired_p) / len(wired_p)
    r = sum(wired_r) / len(wired_r)
    f1 = 2*p*r / (p+r)
    result += 'Total Wired Html files: %d \n' % len(wired_p)
    result += 'Wired Table P: %.2f R: %.2f F1: %.2f \n' % (100*p, 100*r, 100*f1)
    result += "-----------------------------------------------------\n"

    wireless_f1 = evaluate_f1(label_wireless_relations, pred_wireless_relations, num_workers=30)
    wireless_p = [item[0] for item in wireless_f1]
    wireless_r = [item[1] for item in wireless_f1]
    p = sum(wireless_p) / len(wireless_p)
    r = sum(wireless_r) / len(wireless_r)
    f1 = 2*p*r / (p+r)
    result += 'Total Wireless Html files: %d \n' % len(wireless_f1)
    result += 'Wireless Table P: %.2f R: %.2f F1: %.2f \n' % (100*p, 100*r, 100*f1)
    result += "-----------------------------------------------------\n"

    p = wired_p + wireless_p
    r = wired_r + wireless_r
    p = sum(p) / len(p)
    r = sum(r) / len(r)
    f1 = 2*p*r / (p+r)
    result += 'Total Html files: %d \n' % len(wired_p + wireless_p)
    result += 'Table P: %.2f R: %.2f F1: %.2f \n' % (100*p, 100*r, 100*f1)
    result += "-----------------------------------------------------\n"

    with open(os.path.join(html_dir, 'total_result_f1_mp.txt'), 'w') as f:
        f.write(result)
    print('While evaluate %s\n%s' % (html_dir, result))