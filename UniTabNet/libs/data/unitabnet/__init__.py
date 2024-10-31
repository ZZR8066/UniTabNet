import torch
from . import transform as T
from torch.utils.data.distributed import DistributedSampler
from .dataset import Dataset, DataCollator, LRCRecordLoader
from libs.utils.comm import distributed, get_rank, get_world_size


def create_dataset(lrc_paths, image_dirs, config, tokenizer, image_processor):
    loaders = list()
    for lrc_path, image_dir in zip(lrc_paths, image_dirs):
        loader = LRCRecordLoader(lrc_path, image_dir)
        loaders.append(loader)

    transforms = T.Compose([
        T.CallTokenizedInput(512, image_processor, tokenizer),
    ])

    dataset = Dataset(loaders, transforms)
    return dataset