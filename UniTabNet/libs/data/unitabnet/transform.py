import os
import re
import cv2
import copy
import math
import torch
import random
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from libs.data.unitabnet.utils import PhotoMetricDistortion


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, *data):
        for transform in self.transforms:
            data = transform(*data)
        return data


def get_minimum_rotated_rectangle(points):
    """
    使用shapely计算给定点集的最小旋转外接矩形的四个顶点坐标。

    参数:
    points -- 点集，格式为 (x, y) 对的 list。

    返回值:
    一个元组，包含矩形的四个顶点坐标。
    """
    # 创建一个多边形
    poly = Polygon(points)
    
    # 计算最小外接矩形
    mbr = poly.minimum_rotated_rectangle
    
    # 获取最小外接矩形的四个顶点
    x, y = mbr.exterior.coords.xy
    vertices = list(zip(x, y))
    
    # 返回四个顶点坐标（最后一个顶点是重复的，因此去除）
    return vertices[:-1]

    
def sort_vertices(vertices):
    # 计算几何中心
    center = [sum([p[0] for p in vertices]) / len(vertices), sum([p[1] for p in vertices]) / len(vertices)]
    
    # 定义一个计算角度的函数
    def angle_from_center(vertex):
        return math.atan2(vertex[1] - center[1], vertex[0] - center[0])
    
    # 根据相对于中心的角度对顶点进行排序
    sorted_vertices = sorted(vertices, key=angle_from_center)

    return sorted_vertices


class CallTokenizedInput:
    def __init__(self, max_length=768, processor=None, tokenizer=None, num_bins=999):
        self.max_length=max_length
        self.processor=processor
        self.tokenizer=tokenizer
        self.num_bins=num_bins
        self.operation = PhotoMetricDistortion(brightness_delta=8, contrast_range=(0.75, 1.25), saturation_range=(0.9, 1.1), hue_delta=8)

    def __call__(self, info, image):
        image = np.array(image)

        # visualize info['bbox'] for check
        # canvas = copy.deepcopy(image)
        # for poly in info['bbox']:
        #     poly = np.array(poly)
        #     if poly.min() < 0:
        #         continue
        #     cv2.polylines(canvas, [poly.astype(np.int32).reshape(-1,2)], True, (255,0,0), 2)
        # cv2.imwrite('0.png', canvas)

        # visualize info['guider'] for check
        # for poly in info['guider']:
            # canvas = copy.deepcopy(image)
            # cv2.polylines(canvas, [np.array(poly, dtype=np.int32).reshape(-1,2)], True, (255,0,0), 2)
            # cv2.imwrite('0.png', canvas)

        # visualize info['line_poly'] for check
        # canvas = copy.deepcopy(image)
        # for poly in info['line_poly']:
        #     cv2.polylines(canvas, [np.array(poly, dtype=np.int32).reshape(-1,2)], True, (255,0,0), 2)
        # cv2.imwrite('0.png', canvas)

        # image color enchance
        prob = random.random()
        if prob < 0.5:
            image = self.operation(image)
        
        # rotate enhance
        prob = random.random()
        if prob < 0.3:
            angle = random.randrange(-6, 6)
            image, rotation_matrix = self.rotate_image(np.array(image), angle)
        else:
            rotation_matrix = None

        image_info = self.processor(Image.fromarray(image.astype(np.uint8)), random_padding=True, return_tensors="pt")
        pixel_values = image_info.pixel_values[0] # (3, H, W)

        # rescale bboxes
        origin_height, origin_width = image.shape[:2]
        resize_height, resize_width = list(image_info.resized_shape[0].numpy())
        ratio_height, ratio_width = resize_height/origin_height, resize_width/origin_width
        pad_left, pad_top = list(image_info.padding[0].numpy())

        max_span_indexs = 1000 # 1000 is add pos vocab, we share it with span index
        span_eyes_matric = np.eye(max_span_indexs)

        # pretraining task for synthdog
        if 'text_sequence' in info:
            text_sequence = info['text_sequence']
            tokens = self.tokenizer.encode('<text_sequence>'+text_sequence+'</s>', add_special_tokens=False)
            input_ids = tokens[:self.max_length]
            loss_mask = [1] * len(input_ids)

            ocr_input_ids = []
            ocr_loss_mask = []
            accumlate_length = len(input_ids)
            text_polys = info['text_polys']
            texts = info['texts']
            assert len(texts) == len(text_polys)
            random_idx = [idx for idx in range(len(text_polys))]
            random.shuffle(random_idx)
            for idx in random_idx:
                text = texts[idx]
                poly = np.array(text_polys[idx], dtype=np.float32)
                poly = get_minimum_rotated_rectangle(poly.reshape(-1,2))
                poly = np.array(poly).reshape(-1)

                # rotated 
                if rotation_matrix is not None:
                    poly = poly.reshape(-1,2)
                    poly = np.concatenate([poly, np.ones((poly.shape[0], 1))], axis=1)
                    poly = np.dot(rotation_matrix, poly.transpose()).transpose()
                    poly = poly.reshape(-1)

                poly[0::2] = poly[0::2] * ratio_width
                poly[1::2] = poly[1::2] * ratio_height
                poly[0::2] = poly[0::2] + pad_left
                poly[1::2] = poly[1::2] + pad_top

                # quatize
                poly[0::2] = poly[0::2] / self.processor.size['width'] * self.num_bins
                poly[1::2] = poly[1::2] / self.processor.size['height'] * self.num_bins
                poly = poly.clip(0, self.num_bins)

                # sort point
                poly = sort_vertices(poly.reshape(-1,2))
                poly = np.array(poly).reshape(-1).astype(np.int32)

                # # visualize for check
                # image_canvas = np.zeros((self.processor.size['height'], self.processor.size['width'], 3), dtype=np.uint8)
                # resize_image = cv2.resize(image, (resize_width, resize_height))
                # image_canvas[pad_top:pad_top+resize_height, pad_left:pad_left+resize_width] = resize_image
                # poly = np.array(poly, dtype=np.float32).reshape(-1)
                # poly[0::2] = poly[0::2] * self.processor.size['width'] / self.num_bins
                # poly[1::2] = poly[1::2] * self.processor.size['height'] / self.num_bins
                # cv2.polylines(image_canvas, [np.array(poly, dtype=np.int32).reshape(-1,2)], True, (255,0,0), 2)
                # cv2.imwrite('label_polys.png', image_canvas)

                # prompt
                prompt = '<ocr><poly>' + ''.join(['<%d>'%loc for loc in poly]) + '</sep><text>'
                prompt_token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                completion_token_ids = self.tokenizer.encode(text+'</s>', add_special_tokens=False)

                # terminate or not
                sub_input_ids = prompt_token_ids + completion_token_ids
                if accumlate_length + len(sub_input_ids) >= self.max_length-1:
                    break
                
                ocr_loss_mask = ocr_loss_mask + [0] * len(prompt_token_ids) + [1] * len(completion_token_ids)
                ocr_input_ids = ocr_input_ids + sub_input_ids
                accumlate_length += len(sub_input_ids)

            prob = random.random()
            if prob > 0.5:
                input_ids = input_ids + ocr_input_ids
                loss_mask = loss_mask + ocr_loss_mask
            else:
                input_ids = ocr_input_ids + input_ids
                loss_mask = ocr_loss_mask + loss_mask

            label_row_spans = [[-100]*max_span_indexs] * len(input_ids)
            label_col_spans = [[-100]*max_span_indexs] * len(input_ids)
            label_polys = [[-100]*8] * len(input_ids)
            label_guiders = [np.full((self.processor.size['height']//32, self.processor.size['width']//32), -100).reshape(-1).tolist()] * len(input_ids)
        else:
            input_ids = [self.tokenizer.encode('<table_structure_recognition>', add_special_tokens=False)[0]]
            label_row_spans = [[-100]*max_span_indexs]
            label_col_spans = [[-100]*max_span_indexs]
            label_polys = [[-100]*8]
            label_guiders = [np.full((self.processor.size['height']//32, self.processor.size['width']//32), -100).reshape(-1).tolist()]
            for structure, poly, row_span, col_span, guider in zip(info['structure'], info['bbox'], info['row_span'], info['col_span'], info['guider']):
                input_ids.append(self.tokenizer.encode(structure, add_special_tokens=False)[0])

                # process for cell polys
                poly = np.array(poly, dtype=np.float32)
                if poly.max() <= 0 or len(poly) != 4:
                    label_polys.append([-100]*8)
                    label_row_spans.append([-100]*max_span_indexs)
                    label_col_spans.append([-100]*max_span_indexs)
                else:
                    poly = get_minimum_rotated_rectangle(poly)
                    poly = np.array(poly).reshape(-1)
                    # rotated 
                    if rotation_matrix is not None:
                        poly = poly.reshape(-1,2)
                        poly = np.concatenate([poly, np.ones((poly.shape[0], 1))], axis=1)
                        poly = np.dot(rotation_matrix, poly.transpose()).transpose()
                        poly = poly.reshape(-1)

                    poly[0::2] = poly[0::2] * ratio_width
                    poly[1::2] = poly[1::2] * ratio_height
                    poly[0::2] = poly[0::2] + pad_left
                    poly[1::2] = poly[1::2] + pad_top

                    # quatize
                    poly[0::2] = poly[0::2] / self.processor.size['width'] * self.num_bins
                    poly[1::2] = poly[1::2] / self.processor.size['height'] * self.num_bins
                    poly = poly.clip(0, self.num_bins)

                    # sort point
                    poly = sort_vertices(poly.reshape(-1,2))
                    poly = np.array(poly).reshape(-1).astype(np.int64).tolist()
                    label_polys.append(poly)
                    label_row_spans.append(span_eyes_matric[row_span].tolist())
                    label_col_spans.append(span_eyes_matric[col_span].tolist())
                
                # process for guider mask
                if structure not in ["<C>", "<NL>"]:
                    canvas = np.full((self.processor.size['height']//32, self.processor.size['width']//32), -100)
                    label_guiders.append(canvas.reshape(-1).tolist())
                else:
                    canvas = np.zeros((self.processor.size['height']//32, self.processor.size['width']//32))
                    guider = np.array(guider, dtype=np.float32).reshape(-1)
                    # rotated
                    if rotation_matrix is not None:
                        guider = guider.reshape(-1,2)
                        guider = np.concatenate([guider, np.ones((guider.shape[0], 1))], axis=1)
                        guider = np.dot(rotation_matrix, guider.transpose()).transpose()
                        guider = guider.reshape(-1)

                    guider[0::2] = guider[0::2] * ratio_width
                    guider[1::2] = guider[1::2] * ratio_height
                    guider[0::2] = guider[0::2] + pad_left
                    guider[1::2] = guider[1::2] + pad_top

                    # down sample ratio
                    guider = guider / 32.0

                    guider = np.array(guider).astype(np.int32).reshape(-1,1,2)
                    segm = cv2.fillPoly(canvas, [guider], 1)
                    label_guiders.append(segm.reshape(-1).tolist())

            input_ids.append(self.tokenizer.encode('</s>', add_special_tokens=False)[0])
            label_polys.append([-100]*8)
            label_row_spans.append([-100]*max_span_indexs)
            label_col_spans.append([-100]*max_span_indexs)
            label_guiders.append(np.full((self.processor.size['height']//32, self.processor.size['width']//32), -100).reshape(-1).tolist())
            loss_mask = [2] * len(input_ids)
            assert len(input_ids) == len(label_polys) == len(label_guiders)

            # visualize label_poly for check
            # image_canvas = np.zeros((self.processor.size['height'], self.processor.size['width'], 3), dtype=np.uint8)
            # resize_image = cv2.resize(image, (resize_width, resize_height))
            # image_canvas[pad_top:pad_top+resize_height, pad_left:pad_left+resize_width] = resize_image
            # for poly in label_polys:
            #     if min(poly) < 0:
            #         continue
            #     poly = np.array(poly, dtype=np.float32).reshape(-1)
            #     poly[0::2] = poly[0::2] * self.processor.size['width'] / self.num_bins
            #     poly[1::2] = poly[1::2] * self.processor.size['height'] / self.num_bins
            #     cv2.polylines(image_canvas, [np.array(poly, dtype=np.int32).reshape(-1,2)], True, (255,0,0), 2)
            # cv2.imwrite('label_polys.png', image_canvas)

            # visualize segm for check
            # for segm in label_guiders:
            #     if min(segm) < 0:
            #         continue
                # image_canvas = np.zeros((self.processor.size['height'], self.processor.size['width'], 3), dtype=np.uint8)
                # resize_image = cv2.resize(image, (resize_width, resize_height))
                # image_canvas[pad_top:pad_top+resize_height, pad_left:pad_left+resize_width] = resize_image

                # color_mask = np.zeros_like(image_canvas)
                # segm = np.array(segm).reshape(self.processor.size['height']//32, self.processor.size['width']//32)
                # segm = cv2.resize(segm, (self.processor.size['width'], self.processor.size['height']))
                # color_mask[segm==1] = [255,0,0]
                # image_canvas = cv2.addWeighted(image_canvas, 0.7, color_mask, 0.3, 0)

                # cv2.imwrite('guider.png', image_canvas)

            # append table read / ocr task
            prob = random.random()
            if prob > 0.5:
                table_read_input_ids = self.tokenizer.encode('<table_read>'+info['otsl']+'</s>', add_special_tokens=False)
                table_read_label_polys = [[-100]*8] * len(table_read_input_ids)
                table_read_label_row_spans = [[-100]*max_span_indexs] * len(table_read_input_ids)
                table_read_label_col_spans = [[-100]*max_span_indexs] * len(table_read_input_ids)
                table_read_label_guiders = [np.full((self.processor.size['height']//32, self.processor.size['width']//32), -100).reshape(-1).tolist()] * len(table_read_input_ids)
                table_read_loss_mask = [1] * len(table_read_input_ids)

                prob = random.random()
                if prob < 0.9:
                    input_ids = input_ids + table_read_input_ids
                    label_polys = label_polys + table_read_label_polys
                    label_row_spans = label_row_spans + table_read_label_row_spans
                    label_col_spans = label_col_spans + table_read_label_col_spans
                    label_guiders = label_guiders + table_read_label_guiders
                    loss_mask = loss_mask + table_read_loss_mask
                else:
                    input_ids = table_read_input_ids + input_ids
                    label_polys = table_read_label_polys + label_polys
                    label_row_spans = table_read_label_row_spans + label_row_spans
                    label_col_spans = table_read_label_col_spans + label_col_spans
                    label_guiders = table_read_label_guiders + label_guiders
                    loss_mask = table_read_loss_mask + loss_mask
            else:
                text_polys = label_polys[1:-1]
                texts = info['text']
                structures = info['structure']
                random_idx = [idx for idx in range(len(text_polys))]
                random.shuffle(random_idx)
                ocr_input_ids = []
                ocr_loss_mask = []
                accumlate_length = len(input_ids)
                for idx in random_idx:
                    text = texts[idx]
                    structure = structures[idx]
                    if text == '<None>' or len(text) == 0:
                        continue
                    if structure != '<C>':
                        continue
                    text_poly = np.array(text_polys[idx], dtype=np.int64).reshape(-1)

                    # prompt
                    prompt = '<ocr><poly>' + ''.join(['<%d>'%loc for loc in text_poly]) + '</sep><text>'
                    prompt_token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                    completion_token_ids = self.tokenizer.encode(text+'</s>', add_special_tokens=False)

                    # terminate or not
                    sub_input_ids = prompt_token_ids + completion_token_ids
                    if accumlate_length + len(sub_input_ids) >= self.max_length:
                        break
                    
                    ocr_loss_mask = ocr_loss_mask + [0] * len(prompt_token_ids) + [1] * len(completion_token_ids)
                    ocr_input_ids = ocr_input_ids + sub_input_ids
                    accumlate_length += len(sub_input_ids)

                ocr_label_polys = [[-100]*8] * len(ocr_input_ids)
                ocr_label_row_spans = [[-100]*max_span_indexs] * len(ocr_input_ids)
                ocr_label_col_spans = [[-100]*max_span_indexs] * len(ocr_input_ids)
                ocr_label_guiders = [np.full((self.processor.size['height']//32, self.processor.size['width']//32), -100).reshape(-1).tolist()] * len(ocr_input_ids)

                prob = random.random()
                if prob < 0.9:
                    input_ids = input_ids + ocr_input_ids
                    label_polys = label_polys + ocr_label_polys
                    label_row_spans = label_row_spans + ocr_label_row_spans
                    label_col_spans = label_col_spans + ocr_label_col_spans
                    label_guiders = label_guiders + ocr_label_guiders
                    loss_mask = loss_mask + ocr_loss_mask
                else:
                    input_ids = ocr_input_ids + input_ids
                    label_polys = ocr_label_polys + label_polys
                    label_row_spans = ocr_label_row_spans + label_row_spans
                    label_col_spans = ocr_label_col_spans + label_col_spans
                    label_guiders = ocr_label_guiders + label_guiders
                    loss_mask = ocr_loss_mask + loss_mask

            input_ids = input_ids[:self.max_length]
            label_polys = label_polys[:self.max_length]
            label_row_spans = label_row_spans[:self.max_length]
            label_col_spans = label_col_spans[:self.max_length]
            label_guiders = label_guiders[:self.max_length]
            loss_mask = loss_mask[:self.max_length]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(label_polys, dtype=torch.long), \
            torch.tensor(label_row_spans, dtype=torch.float32), torch.tensor(label_col_spans, dtype=torch.float32), \
                torch.tensor(label_guiders, dtype=torch.float32), pixel_values, torch.tensor(loss_mask, dtype=torch.float)

    def rotate_image(self, image, angle):
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int(height * sin + width * cos)
        new_height = int(height * cos + width * sin)
        rotation_matrix[0, 2] += (new_width - width) / 2
        rotation_matrix[1, 2] += (new_height - height) / 2
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return rotated_image, rotation_matrix