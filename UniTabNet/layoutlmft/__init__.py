from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers import TOKENIZER_MAPPING
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, BertConverter

from .models.unitabnet import (
    UniTabNetConfig,
    UniTabNetModel,
)

AutoConfig.register("UniTabNet", UniTabNetConfig)
AutoModel.register(UniTabNetConfig, UniTabNetModel)
