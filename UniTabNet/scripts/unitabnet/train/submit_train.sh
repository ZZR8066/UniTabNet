#! /bin/bash

ky exp submit PtJob \
     -a zrzhang6 \
     -d "unitable-v36-zrzhang6" \
     --experimentName Pretrain \
     --modelName vdu1.1.0.6.1-$(date +%Y%m%d_%H%M) \
     --modelPath '/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/code/mdocp-v2/scripts/gma' \
     -i reg.deeplearning.cn/dlaas/cv_dist:0.1 \
     -e train.sh \
     -l train.log \
     -o train.err \
     --useGpu \
     -g 6 \
     --useDist \
     -w 1 \
     --proID 56529 \
     -k TeslaV100-PCIE-48GB \
     -r cogllm-public \
     --weChatOnStatus