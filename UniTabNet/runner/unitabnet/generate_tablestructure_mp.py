import os
import cv2
import sys
import tqdm
import json
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
from PIL import Image
from libs.data.unitabnet import LRCRecordLoader
from layoutlmft.models.unitabnet import UniTabNetModel
from transformers import DonutProcessor, AutoTokenizer
import torch
import subprocess
import numpy as np
from transformers import AutoConfig
from libs.data.unitabnet.utils import ImageProcessor
from utils import str2image
from libs.eval.format_translate import table_to_html, format_html
from libs.eval.metric import TEDSMetric


def str2bool(v):
    return v.lower() in ("true", "t", "1")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--lrc_path', type=str, default=None)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=int(1e8))
    parser.add_argument("--use_mp", type=str2bool, default=False)
    parser.add_argument('--process_id', type=int, default=0)
    parser.add_argument('--total_process_num', type=int, default=1)
    args = parser.parse_args()

    return args


def single_process(args):
    
    # model path & output_dir
    processor_path = '/train27/mmu/permanent/zrzhang6/TableQA/pretrained_model/donut-based-tsr-init/'
    model_path = args.model_path
    output_dir = args.output_dir

    # load document image
    image_dir = args.image_dir
    lrc_path = args.lrc_path
    loader = LRCRecordLoader(lrc_path, image_dir)
    num_bins = 999

    # valid idxs
    valid_ids = list(range(len(loader)))[args.start_index:args.end_index]
    valid_ids = valid_ids[args.process_id::args.total_process_num]

    # init processor
    tokenizer = DonutProcessor.from_pretrained(processor_path).tokenizer
    image_processor = ImageProcessor.from_pretrained(processor_path)
    image_processor.size = dict(height=1600, width=1600)

    # init model
    model = UniTabNetModel.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for idx in tqdm.tqdm(valid_ids):
        info, image = loader.get_data(idx)

        image_info = image_processor(Image.fromarray(image.astype(np.uint8)), random_padding=False, return_tensors="pt")
        pixel_values = image_info.pixel_values # (1, 3, H, W)

        origin_height, origin_width = image.shape[:2]
        resize_height, resize_width = list(image_info.resized_shape[0].numpy())
        ratio_height, ratio_width = resize_height/origin_height, resize_width/origin_width
        pad_left, pad_top = list(image_info.padding[0].numpy())

        # prepare decoder inputs
        decoder_input_ids = tokenizer('<s><table_structure_recognition>', add_special_tokens=False, return_tensors="pt").input_ids

        outputs = model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=500,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[tokenizer.unk_token_id]],
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        token_ids = outputs.sequences.cpu().numpy().tolist()[0]

        # decoder hidden state
        decoder_hidden_states = []
        for hidden_state in outputs.decoder_hidden_states:
            decoder_hidden_states.append(hidden_state[-1])
        decoder_hidden_states = torch.cat(decoder_hidden_states, dim=1)

        # box predict
        pos_key = model.decoder.model.decoder.embed_tokens.weight[-1000:][None]
        pos_x1_query = model.poly_x1_proj(decoder_hidden_states)
        pos_x1_logit = torch.bmm(pos_x1_query, pos_key.transpose(1, 2))
        pos_y1_query = model.poly_y1_proj(decoder_hidden_states)
        pos_y1_logit = torch.bmm(pos_y1_query, pos_key.transpose(1, 2))
        pos_x2_query = model.poly_x2_proj(decoder_hidden_states)
        pos_x2_logit = torch.bmm(pos_x2_query, pos_key.transpose(1, 2))
        pos_y2_query = model.poly_y2_proj(decoder_hidden_states)
        pos_y2_logit = torch.bmm(pos_y2_query, pos_key.transpose(1, 2))
        pos_x3_query = model.poly_x3_proj(decoder_hidden_states)
        pos_x3_logit = torch.bmm(pos_x3_query, pos_key.transpose(1, 2))
        pos_y3_query = model.poly_y3_proj(decoder_hidden_states)
        pos_y3_logit = torch.bmm(pos_y3_query, pos_key.transpose(1, 2))
        pos_x4_query = model.poly_x4_proj(decoder_hidden_states)
        pos_x4_logit = torch.bmm(pos_x4_query, pos_key.transpose(1, 2))
        pos_y4_query = model.poly_y4_proj(decoder_hidden_states)
        pos_y4_logit = torch.bmm(pos_y4_query, pos_key.transpose(1, 2))

        pos_logit = torch.stack((pos_x1_logit, pos_y1_logit, pos_x2_logit, pos_y2_logit, \
            pos_x3_logit, pos_y3_logit, pos_x4_logit, pos_y4_logit), dim=2)
        
        pos_logit = pos_logit.softmax(-1)
        pos_range = torch.arange(pos_logit.shape[-1], dtype=pos_logit.dtype, device=pos_logit.device)[None, None, None]
        pos_logit = (pos_range * pos_logit).sum(-1) # (B,L,8)
        polys = pos_logit[0].data.cpu().numpy().tolist()

        # row spans predict
        row_span_key = model.decoder.model.decoder.embed_tokens.weight[-1000:][None]
        row_span_query = model.row_span_proj(decoder_hidden_states)
        row_span_logit = torch.bmm(row_span_query, row_span_key.transpose(1,2))
        row_spans = row_span_logit.argmax(-1)[0].cpu().numpy().tolist()

        # col spans predict
        col_span_key = model.decoder.model.decoder.embed_tokens.weight[-1000:][None]
        col_span_query = model.col_span_proj(decoder_hidden_states)
        col_span_logit = torch.bmm(col_span_query, col_span_key.transpose(1,2))
        col_spans = col_span_logit.argmax(-1)[0].cpu().numpy().tolist()

        # rescale predict polys to origin image size
        pt_cell_polys = np.array(polys, dtype=np.float32).reshape(-1,8)
        pt_cell_polys[:, 0::2] = pt_cell_polys[:, 0::2] * image_processor.size['width'] / num_bins
        pt_cell_polys[:, 1::2] = pt_cell_polys[:, 1::2] * image_processor.size['height'] / num_bins
        pt_cell_polys[:, 0::2] = pt_cell_polys[:, 0::2] - pad_left
        pt_cell_polys[:, 1::2] = pt_cell_polys[:, 1::2] - pad_top
        pt_cell_polys[:, 0::2] = pt_cell_polys[:, 0::2] / ratio_width
        pt_cell_polys[:, 1::2] = pt_cell_polys[:, 1::2] / ratio_height
        pt_cell_polys = pt_cell_polys.clip(0, 1e5)
        pt_cell_polys = pt_cell_polys.reshape(-1, 4, 2)
        pt_cell_polys = pt_cell_polys.tolist()
        pred_structures = [tokenizer.decode(item) for item in token_ids[1:]]

        for index in range(len(pred_structures)):
            if pred_structures[index] != '<C>':
                pt_cell_polys[index] = [[0]*2]*4

        line_polys = []
        text_contents = []
        for poly, text in zip(info['bbox'], info['text']):
            if np.array(poly).max() <= 0:
                continue
            if text == '<None>' or len(text) == 0:
                continue
            line_polys.append(poly)
            text_contents.append(text)

        pred_html = table_to_html(pred_structures, row_spans, col_spans, pt_cell_polys, line_polys, text_contents)
        pred_html['format_html'] = format_html(pred_html)
        pred_html['pt_cell_polys'] = pt_cell_polys
        pred_html['pred_structures'] = pred_structures
        with open(os.path.join(output_dir, '%04d_pred_html.json' % idx), 'w') as f:
            json.dump(pred_html, f, indent=4)

        # gt cell polys
        gt_cell_polys = []
        for poly in info['bbox']:
            poly = np.array(poly)
            if poly.max() <= 0:
                gt_cell_polys.append([[0]*2]*4)
            else:
                gt_cell_polys.append(poly.tolist())

        label_html = table_to_html(info['structure'], info['row_span'], info['col_span'], gt_cell_polys, line_polys, text_contents)
        label_html['format_html'] = format_html(label_html)
        label_html['html']['line_poly'] = line_polys
        label_html['html']['text_contents'] = text_contents
        label_html['image_idx'] = idx
        label_html['html']['image_path'] = os.path.join(image_dir, os.path.basename(info['image_path']))
        with open(os.path.join(output_dir, '%04d_label_html.json' % idx), 'w') as f:
            json.dump(label_html, f, indent=4)


def main():
    args = parse_args()
    if not args.use_mp:
        single_process(args)
    else:
        p_list = []
        total_process_num = args.total_process_num
        for process_id in range(total_process_num):
            cmd = [sys.executable, "-u"] + sys.argv + [
                "--process_id={}".format(process_id),
                "--use_mp={}".format(False)
            ]
            p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
            p_list.append(p)
        for p in p_list:
            p.wait()


if __name__ == "__main__":
    main()