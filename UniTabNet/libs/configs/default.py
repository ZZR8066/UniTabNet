from libs.utils.counter import Counter

# train path
train_syn_zh_lrc_paths = [\
    "/yrfs2/cv1/jszhang6/zrzhang6/MultiModelMathAnswer/dataprocess/process_synthdog/output/train-1/info.lrc", \
    "/yrfs2/cv1/jszhang6/zrzhang6/MultiModelMathAnswer/dataprocess/process_synthdog/output/train-2/info.lrc", \
    "/yrfs2/cv1/jszhang6/zrzhang6/MultiModelMathAnswer/dataprocess/process_synthdog/output/train-3/info.lrc", \
    "/yrfs2/cv1/jszhang6/zrzhang6/MultiModelMathAnswer/dataprocess/process_synthdog/output/train-4/info.lrc", \
    "/yrfs2/cv1/jszhang6/zrzhang6/MultiModelMathAnswer/dataprocess/process_synthdog/output/train-5/info.lrc"
]

train_syn_zh_image_dirs = [\
    "/yrfs2/cv1/jszhang6/zrzhang6/SVRD-Competition/code/donut/synthdog/outputs/synthdog_part2/zh/1/train", \
    "/yrfs2/cv1/jszhang6/zrzhang6/SVRD-Competition/code/donut/synthdog/outputs/synthdog_part2/zh/2/train", \
    "/yrfs2/cv1/jszhang6/zrzhang6/SVRD-Competition/code/donut/synthdog/outputs/synthdog_part2/zh/3/train", \
    "/yrfs2/cv1/jszhang6/zrzhang6/SVRD-Competition/code/donut/synthdog/outputs/synthdog_part2/zh/4/train", \
    "/yrfs2/cv1/jszhang6/zrzhang6/SVRD-Competition/code/donut/synthdog/outputs/synthdog_part2/zh/5/train", \
]

# valid path
valid_syn_zh_lrc_paths = [\
    "/yrfs2/cv1/jszhang6/zrzhang6/MultiModelMathAnswer/dataprocess/process_synthdog/output/validation-1/info.lrc"
]

valid_syn_zh_image_dirs = [\
    "/yrfs2/cv1/jszhang6/zrzhang6/SVRD-Competition/code/donut/synthdog/outputs/synthdog_part2/zh/1/validation/"
]


# dataset path dict
datasets = dict(
    # train
    train_syn_zh_lrc_paths=train_syn_zh_lrc_paths,
    train_syn_zh_image_dirs=train_syn_zh_image_dirs,

    # valid
    valid_syn_zh_lrc_paths=valid_syn_zh_lrc_paths,
    valid_syn_zh_image_dirs=valid_syn_zh_image_dirs
)

# counter for show each item loss
counter = Counter(cache_nums=50)