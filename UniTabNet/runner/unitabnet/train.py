#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys

import numpy as np

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

import libs.configs.default as cfg
from libs.data.unitabnet import create_dataset, DataCollator

import torch
import transformers
from layoutlmft.data.data_args import DataTrainingArguments
from layoutlmft.models.model_args import ModelArguments
from runner.unitabnet.trainer import PreTrainer as Trainer
from transformers import (
    AutoTokenizer,
    AutoConfig,
    DonutProcessor,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from layoutlmft.models.unitabnet import UniTabNetConfig, UniTabNetModel
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from libs.data.unitabnet.utils import ImageProcessor


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in layoutlmft/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    processor = DonutProcessor.from_pretrained(model_args.model_name_or_path)
    tokenizer = processor.tokenizer
    image_processor = ImageProcessor.from_pretrained(model_args.model_name_or_path)
    image_processor.size = dict(height=1600, width=1600)

    added_tsr_tokens=['<C>','<U>','<X>','<L>','<NL>','<None>']
    for token in added_tsr_tokens:
        assert token in processor.tokenizer.get_added_vocab()


    pretrain_model_path = "/train21/mmu/permanent/zrzhang6/TableQA/experiments/unitable-v26/pretrain/checkpoint-160000/"
    config = UniTabNetConfig.from_pretrained(pretrain_model_path)
    model = UniTabNetModel.from_pretrained(pretrain_model_path)

    # special token id for model decoder
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Initialize pretrain dataset and valid dataset
    synthdog_en_train_lrc_path =[
        "/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/dataprocess/process_synthdog/output/tableqa/en/train-1/info.lrc",
        "/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/dataprocess/process_synthdog/output/tableqa/en/train-2/info.lrc",
        "/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/dataprocess/process_synthdog/output/tableqa/en/train-3/info.lrc",
        "/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/dataprocess/process_synthdog/output/tableqa/en/train-4/info.lrc",
        "/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/dataprocess/process_synthdog/output/tableqa/en/train-5/info.lrc",
    ]
    synthdog_en_train_image_dir = [
        "/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/copy2rdg/datasets/tableqa/image/",
        "/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/copy2rdg/datasets/tableqa/image/",
        "/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/copy2rdg/datasets/tableqa/image/",
        "/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/copy2rdg/datasets/tableqa/image/",
        "/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/copy2rdg/datasets/tableqa/image/",
    ]
    synthdog_zh_train_lrc_path = [
        "/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/dataprocess/process_synthdog/output/tableqa/zh/train-1/info.lrc",
        "/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/dataprocess/process_synthdog/output/tableqa/zh/train-2/info.lrc",
        "/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/dataprocess/process_synthdog/output/tableqa/zh/train-3/info.lrc",
        "/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/dataprocess/process_synthdog/output/tableqa/zh/train-4/info.lrc",
        "/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/dataprocess/process_synthdog/output/tableqa/zh/train-5/info.lrc",
    ]
    synthdog_zh_train_image_dir = [
        "/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/copy2rdg/datasets/tableqa/image/",
        "/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/copy2rdg/datasets/tableqa/image/",
        "/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/copy2rdg/datasets/tableqa/image/",
        "/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/copy2rdg/datasets/tableqa/image/",
        "/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/copy2rdg/datasets/tableqa/image/",
    ]

    iflytab_wired_train_lrc_path = ["/train27/mmu/permanent/zrzhang6/TableQA/datasets/compress_iflytab_wired/train/info.lrc"] * 5
    iflytab_wired_train_image_dir = ["/train27/mmu/permanent/zrzhang6/TableQA/datasets/compress_iflytab_wired/images"] * 5
    iflytab_wireless_train_lrc_path = ["/train27/mmu/permanent/zrzhang6/TableQA/datasets/compress_iflytab_wireless/train/info.lrc"] * 5
    iflytab_wireless_train_image_dir = ["/train27/mmu/permanent/zrzhang6/TableQA/datasets/compress_iflytab_wireless/images"] * 5
    pubtables_train_lrc_path = ["/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/copy2rdg/datasets/tableqa/withguider/PubTables_train_infos.lrc"]
    pubtables_traim_image_dir = ["/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/copy2rdg/datasets/tableqa/image/"]

    iflytab_wired_crop_train_lrc_path = ["/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/copy2rdg/datasets/tableqa/iFLYTAB_crop_aug/lrc/iFLYTAB_train_aug_wired_infos.lrc"]
    iflytab_wired_crop_train_image_dir = ["/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/copy2rdg/datasets/tableqa/iFLYTAB_crop_aug/image/wired"]
    iflytab_wireless_crop_train_lrc_path = ["/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/copy2rdg/datasets/tableqa/iFLYTAB_crop_aug/lrc/iFLYTAB_train_aug_wireless_infos.lrc"]
    iflytab_wireless_crop_train_image_dir = ["/train21/mmu/permanent/zrzhang6/OCRMultiModalLLM/copy2rdg/datasets/tableqa/iFLYTAB_crop_aug/image/wireless"]


    train_lrc_paths = pubtables_train_lrc_path
    train_image_dirs = pubtables_traim_image_dir

    valid_lrc_paths=["/train27/mmu/permanent/zrzhang6/TableQA/datasets/compress_iflytab/info.lrc"]
    valid_image_dirs=['/train27/mmu/permanent/zrzhang6/TableQA/datasets/compress_iflytab/images/']

    train_dataset = create_dataset(train_lrc_paths, train_image_dirs, config, tokenizer, image_processor)
    valid_dataset = create_dataset(valid_lrc_paths, valid_image_dirs, config, tokenizer, image_processor)
    data_collator = DataCollator(tokenizer)

    # Metrics
    class compute_metrics:
        def __init__(self):
            pass
        
        def __call__(self, p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)
            
            mask = labels > -1
            pred = np.logical_and(predictions==labels, mask)
            correct_nums = pred.sum()
            total_nums = max(mask.sum(), 1e-6)
            
            acc = correct_nums / total_nums

            results = {
                "precision": acc,
                "recall": acc,
                "f1": acc,
                "accuracy": acc,
            }

            logger.info("***** Eval results *****")
            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, str(results[key]))

            return results

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=valid_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
