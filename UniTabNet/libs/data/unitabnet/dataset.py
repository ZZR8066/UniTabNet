import os
import cv2
import torch
import random
from .list_record_cache import ListRecordLoader


class InvalidFormat(Exception):
    pass


class LRCRecordLoader:
    def __init__(self, lrc_path, image_dir):
        self.image_dir = image_dir
        self.loader = ListRecordLoader(lrc_path)

    def __len__(self):
        return len(self.loader)

    def get_data(self, idx):
        info = self.loader.get_record(idx)
        if 'text_sequence' in info: # Synthdog
            image = cv2.imread(info['image_path'])
        else:
            image_name = os.path.basename(info['image_path'])
            image = cv2.imread(os.path.join(self.image_dir, image_name))
        return info, image


class Dataset:
    def __init__(self, loaders, transforms):
        self.loaders = loaders
        self.transforms = transforms

    def _match_loader(self, idx):
        offset = 0
        for loader in self.loaders:
            if len(loader) + offset > idx:
                return loader, idx - offset
            else:
                offset += len(loader)
        raise IndexError()

    def get_info(self, idx):
        loader, rela_idx = self._match_loader(idx)
        return loader.get_info(rela_idx)

    def __len__(self):
        return sum([len(loader) for loader in self.loaders])

    def __getitem__(self, idx):
        try:
            loader, rela_idx = self._match_loader(idx)
            info, image = loader.get_data(rela_idx)
            input_ids, label_polys, label_row_spans, label_col_spans, label_guiders, pixel_values, loss_mask = self.transforms(info, image)
            return dict(
                labels=input_ids,
                label_polys=label_polys,
                label_row_spans=label_row_spans,
                label_col_spans=label_col_spans,
                label_guiders=label_guiders,
                pixel_values=pixel_values,
                loss_mask=loss_mask
            )
        except Exception as e:
            print('Error occured while load data: %d' % idx)
            raise e
        

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch_data):

        def merge1d(tensors, pad_id):
            lengths= [len(s) for s in tensors]
            out = tensors[0].new(len(tensors), max(lengths)).fill_(pad_id)
            for i, s in enumerate(tensors):
                out[i,:len(s)] = s
            return out
    
        def merge2d(tensors, pad_id):
            dim1 = max([s.shape[0] for s in tensors])
            dim2 = max([s.shape[1] for s in tensors])
            out = tensors[0].new(len(tensors), dim1, dim2).fill_(pad_id)
            for i, s in enumerate(tensors):
                out[i, :s.shape[0], :s.shape[1]] = s
            return out

        def merge3d(tensors, pad_id):
            dim1 = max([s.shape[0] for s in tensors])
            dim2 = max([s.shape[1] for s in tensors])
            dim3 = max([s.shape[2] for s in tensors])
            out = tensors[0].new(len(tensors), dim1, dim2, dim3).fill_(pad_id)
            for i, s in enumerate(tensors):
                out[i, :s.shape[0], :s.shape[1], :s.shape[2]] = s
            return out

        def mask1d(tensors, pad_id):
            lengths= [len(s) for s in tensors]
            out = tensors[0].new(len(tensors), max(lengths)).fill_(pad_id)
            for i, s in enumerate(tensors):
                out[i,:len(s)] = 1
            return out

        pixel_values = merge3d([data["pixel_values"] for data in batch_data], 0)
        labels = merge1d([data['labels'] for data in batch_data], -100) # -100 is ignore idx
        label_polys = merge2d([data['label_polys'] for data in batch_data], -100) # -100 is ignore ids
        label_row_spans = merge2d([data['label_row_spans'] for data in batch_data], -100) # -100 is ignore ids
        label_col_spans = merge2d([data['label_col_spans'] for data in batch_data], -100) # -100 is ignore ids
        label_guiders = merge2d([data['label_guiders'] for data in batch_data], -100) # -100 is ignore ids
        loss_mask = merge1d([data['loss_mask'] for data in batch_data], 0)

        return {
            "pixel_values":pixel_values,
            "labels":labels,
            "label_polys": label_polys,
            "label_row_spans": label_row_spans,
            "label_col_spans": label_col_spans,
            "label_guiders": label_guiders,
            "loss_mask":loss_mask,
            "return_dict":False
        }