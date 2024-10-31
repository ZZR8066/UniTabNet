import json
import torch
import os
import sys

import numpy as np

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
from layoutlmft.models.unitabnet import UniTabNetConfig, UniTabNetModel
from transformers import AutoTokenizer, AutoConfig

origin_model_path = '/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/copy2rdg/datasets/tableqa/donut-base-finetuned-zhtrainticket-tsr/'
modified_model_path = '/train27/mmu/permanent/zrzhang6/TableQA/pretrained_model/donut-based-tsr-init/'
print("tokenizer load from ", origin_model_path)
tokenizer = AutoTokenizer.from_pretrained(origin_model_path)

# 1. add task specifical token
add_dict = dict()
add_dict['<text_sequence>'] = tokenizer.vocab_size + 0
add_dict['<table_read>'] = tokenizer.vocab_size + 1
add_dict['<ocr>'] = tokenizer.vocab_size + 2
add_dict['<vqa>'] = tokenizer.vocab_size + 3
add_dict['<question>'] = tokenizer.vocab_size + 4
add_dict['<answer>'] = tokenizer.vocab_size + 5
add_dict['<poly>'] = tokenizer.vocab_size + 6
add_dict['<bbox>'] = tokenizer.vocab_size + 7
add_dict['<C>'] = tokenizer.vocab_size + 8
add_dict['<L>'] = tokenizer.vocab_size + 9
add_dict['<U>'] = tokenizer.vocab_size + 10
add_dict['<X>'] = tokenizer.vocab_size + 11
add_dict['<NL>'] = tokenizer.vocab_size + 12
add_dict['<None>'] = tokenizer.vocab_size + 13
add_dict['</sep>'] = tokenizer.vocab_size + 14
add_dict['<tmp-8>'] = tokenizer.vocab_size + 15
add_dict['<tmp-9>'] = tokenizer.vocab_size + 16
add_dict['<tmp-10>'] = tokenizer.vocab_size + 17
add_dict['<tmp-11>'] = tokenizer.vocab_size + 18
add_dict['<tmp-12>'] = tokenizer.vocab_size + 19
add_dict['<tmp-13>'] = tokenizer.vocab_size + 20
add_dict['<tmp-14>'] = tokenizer.vocab_size + 21
add_dict['<tmp-15>'] = tokenizer.vocab_size + 22
add_dict['<tmp-16>'] = tokenizer.vocab_size + 23
add_dict['<tmp-17>'] = tokenizer.vocab_size + 24
add_dict['<tmp-18>'] = tokenizer.vocab_size + 25

# add pos token
shift_pos = 26
pos_range = 1000
for pos in range(pos_range):
    key = '<%d>' % pos
    add_dict[key] = tokenizer.vocab_size + shift_pos + pos 

# asset repeat in origin vocab
vocab_key = list(tokenizer.vocab)
# for key in list(add_dict.keys()):
#     assert key not in vocab_key

with open(os.path.join(modified_model_path, 'added_tokens.json'), 'w') as json_file:
    json.dump(add_dict, json_file, indent=4)

# modify checkpoint
new_tokenizer = AutoTokenizer.from_pretrained(modified_model_path)
donut_config = AutoConfig.from_pretrained(origin_model_path)
config_encoder = donut_config.encoder
config_decoder = donut_config.decoder
config_decoder.max_position_embeddings *= 3
config_decoder.vocab_size = len(new_tokenizer.vocab)
new_tokenizer = AutoTokenizer.from_pretrained(modified_model_path)
config = UniTabNetConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
model = UniTabNetModel(config=config)

# init model 
checkpoint = torch.load(os.path.join(origin_model_path, 'pytorch_model.bin'), map_location='cpu')
tmp = checkpoint['decoder.model.decoder.embed_positions.weight']
checkpoint['decoder.model.decoder.embed_positions.weight'] = model.decoder.model.decoder.embed_positions.weight.data
checkpoint['decoder.model.decoder.embed_positions.weight'][:tmp.shape[0]] = tmp
tmp = checkpoint['decoder.model.decoder.embed_tokens.weight']
checkpoint['decoder.model.decoder.embed_tokens.weight'] = model.decoder.model.decoder.embed_tokens.weight.data
checkpoint['decoder.model.decoder.embed_tokens.weight'][:tmp.shape[0]] = tmp
checkpoint['decoder.lm_head.weight'] = checkpoint['decoder.model.decoder.embed_tokens.weight']
info = model.load_state_dict(checkpoint, strict=False)
torch.save(checkpoint, os.path.join(modified_model_path, 'pytorch_model.bin'))
print('while init model ', info)
