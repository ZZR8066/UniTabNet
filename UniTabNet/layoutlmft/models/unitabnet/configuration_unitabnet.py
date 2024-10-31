# coding=utf-8
from transformers.utils import logging

from transformers.models.vision_encoder_decoder.configuration_vision_encoder_decoder import VisionEncoderDecoderConfig
from typing import Any, List, Optional, Union
import os
logger = logging.get_logger(__name__)


class UniTabNetConfig(VisionEncoderDecoderConfig):
    model_type = "UniTabNet"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)



