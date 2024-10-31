import torch.nn.functional as F
from libs.configs.default import counter
from .configuration_unitabnet import UniTabNetConfig
from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import *

from transformers.models.donut.modeling_donut_swin import *
logger = logging.get_logger(__name__)


def sigmoid_focal_loss(pred,
                       target,
                       weight=1.0,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean'):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    weight = (alpha * target + (1 - alpha) * (1 - target)) * weight
    weight = weight * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * weight
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super().__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class UniTabNetModel(PreTrainedModel):
    config_class = UniTabNetConfig
    base_model_prefix = "UniTabNet"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    ):
        if config is None and (encoder is None or decoder is None):
            raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
        if config is None:
            config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal"
                    f" to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
                    f" `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for"
                    " `config.encoder.hidden_size`."
                )

        # initialize with config
        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        super().__init__(config)

        if encoder is None:
            encoder = AutoModel.from_config(config.encoder)

        if decoder is None:
            decoder = AutoModelForCausalLM.from_config(config.decoder)

        self.encoder = encoder
        self.decoder = decoder

        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config:"
                f" {self.config.encoder}"
            )
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config:"
                f" {self.config.decoder}"
            )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

        # encoder outputs might need to be projected to different dimension for decoder
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            self.enc_to_dec_proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)

        if self.encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
            )

        self.encoder_pos_emb = PositionalEmbedding(self.decoder.config.hidden_size)
        
        self.poly_x1_proj = nn.Linear(self.decoder.config.hidden_size, self.decoder.config.hidden_size)
        self.poly_y1_proj = nn.Linear(self.decoder.config.hidden_size, self.decoder.config.hidden_size)
        self.poly_x2_proj = nn.Linear(self.decoder.config.hidden_size, self.decoder.config.hidden_size)
        self.poly_y2_proj = nn.Linear(self.decoder.config.hidden_size, self.decoder.config.hidden_size)
        self.poly_x3_proj = nn.Linear(self.decoder.config.hidden_size, self.decoder.config.hidden_size)
        self.poly_y3_proj = nn.Linear(self.decoder.config.hidden_size, self.decoder.config.hidden_size)
        self.poly_x4_proj = nn.Linear(self.decoder.config.hidden_size, self.decoder.config.hidden_size)
        self.poly_y4_proj = nn.Linear(self.decoder.config.hidden_size, self.decoder.config.hidden_size)

        self.row_span_proj = nn.Linear(self.decoder.config.hidden_size, self.decoder.config.hidden_size)
        self.col_span_proj = nn.Linear(self.decoder.config.hidden_size, self.decoder.config.hidden_size)
        self.guider_proj = nn.Linear(self.decoder.config.hidden_size, self.decoder.config.hidden_size)

        self.awl = AutomaticWeightedLoss(5)

    def _set_gradient_checkpointing(self, module, value=False):
        # call both encoder and decoder function on gradient checkpointing
        self.encoder._set_gradient_checkpointing(module, value=value)
        self.decoder._set_gradient_checkpointing(module, value=value)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer
        >>> from PIL import Image
        >>> import requests

        >>> image_processor = AutoImageProcessor.from_pretrained("ydshieh/vit-gpt2-coco-en")
        >>> decoder_tokenizer = AutoTokenizer.from_pretrained("ydshieh/vit-gpt2-coco-en")
        >>> model = VisionEncoderDecoderModel.from_pretrained("ydshieh/vit-gpt2-coco-en")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> img = Image.open(requests.get(url, stream=True).raw)
        >>> pixel_values = image_processor(images=img, return_tensors="pt").pixel_values  # Batch size 1

        >>> output_ids = model.generate(
        ...     pixel_values, max_length=16, num_beams=4, return_dict_in_generate=True
        ... ).sequences

        >>> preds = decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        >>> preds = [pred.strip() for pred in preds]

        >>> assert preds == ["a cat laying on top of a couch next to another cat"]
        ```"""

        from_tf = kwargs.pop("from_tf", False)
        if from_tf:
            from transformers import TFVisionEncoderDecoderModel

            # a workaround to load from tensorflow checkpoint
            # Using `_tf_model` won't work, because the weight names in the encoder/decoder of `_tf_model` get
            # extended before saving those components. For example, The name of `_tf_model.encoder.vit` is
            # `[top model name]/encoder/vit`, but the name of `tf_model.encoder.vit` is `[top model name]/vit`. The
            # [top model name] is handled (stripped) by the conversion method, and the former case gets extra `encoder`,
            # which should not occur when we want to save the components alone.
            # There was a (very) ugly potential fix, which wasn't integrated to `transformers`: see
            #   https://github.com/huggingface/transformers/pull/13222/commits/dbb3c9de76eee235791d2064094654637c99f36d#r697304245
            #   (the change in `src/transformers/modeling_tf_utils.py`)
            _tf_model = TFVisionEncoderDecoderModel.from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
            config = _tf_model.config

            # Using `tf_model` instead
            encoder = _tf_model.encoder.__class__(_tf_model.config.encoder)
            decoder = _tf_model.decoder.__class__(_tf_model.config.decoder)
            # Make sure models are built
            encoder(encoder.dummy_inputs)
            decoder(decoder.dummy_inputs)

            # Get the variable correspondence between `_tf_model` and `encoder` and `decoder`
            encoder_variables = {}
            for v in encoder.trainable_variables + encoder.non_trainable_variables:
                encoder_variables["/".join(v.name.split("/")[1:])] = v
            decoder_variables = {}
            for v in decoder.trainable_variables + decoder.non_trainable_variables:
                decoder_variables["/".join(v.name.split("/")[1:])] = v

            _encoder_variables = {}
            for v in _tf_model.encoder.trainable_variables + _tf_model.encoder.non_trainable_variables:
                _encoder_variables["/".join(v.name.split("/")[2:])] = v
            _decoder_variables = {}
            for v in _tf_model.decoder.trainable_variables + _tf_model.decoder.non_trainable_variables:
                _decoder_variables["/".join(v.name.split("/")[2:])] = v

            # assign weight values to `encoder` and `decoder` from `_tf_model`
            for name, v in encoder_variables.items():
                v.assign(_encoder_variables[name])
            for name, v in decoder_variables.items():
                v.assign(_decoder_variables[name])

            tf_model = TFVisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

            # Deal with `enc_to_dec_proj`
            if hasattr(_tf_model, "enc_to_dec_proj"):
                tf_model(tf_model.dummy_inputs)
                tf_model.enc_to_dec_proj.kernel.assign(_tf_model.enc_to_dec_proj.kernel)
                tf_model.enc_to_dec_proj.bias.assign(_tf_model.enc_to_dec_proj.bias)

            with tempfile.TemporaryDirectory() as tmpdirname:
                encoder_dir = os.path.join(tmpdirname, "encoder")
                decoder_dir = os.path.join(tmpdirname, "decoder")
                tf_model.encoder.save_pretrained(encoder_dir)
                tf_model.decoder.save_pretrained(decoder_dir)

                if hasattr(tf_model, "enc_to_dec_proj"):
                    enc_to_dec_proj_weight = torch.transpose(
                        torch.from_numpy(tf_model.enc_to_dec_proj.kernel.numpy()), 1, 0
                    )
                    enc_to_dec_proj_bias = torch.from_numpy(tf_model.enc_to_dec_proj.bias.numpy())

                del _tf_model
                del tf_model
                gc.collect()

                model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
                    encoder_dir, decoder_dir, encoder_from_tf=True, decoder_from_tf=True
                )
                # This is only for copying some specific attributes of this particular model.
                model.config = config

                if hasattr(model, "enc_to_dec_proj"):
                    model.enc_to_dec_proj.weight.data = enc_to_dec_proj_weight
                    model.enc_to_dec_proj.bias.data = enc_to_dec_proj_bias

                return model

        # At the moment fast initialization is not supported for composite models
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for VisionEncoderDecoderModel. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False

        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        r"""
        Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model
        checkpoints.


        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you need to first set it back in training mode with `model.train()`.

        Params:
            encoder_pretrained_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the image encoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co. An
                      example is `google/vit-base-patch16-224-in21k`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            decoder_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the text decoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args (remaining positional arguments, *optional*):
                All remaning positional arguments will be passed to the underlying model's `__init__` method.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`).

                - To update the encoder configuration, use the prefix *encoder_* for each configuration parameter.
                - To update the decoder configuration, use the prefix *decoder_* for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a `config` is provided or automatically loaded.

        Example:

        ```python
        >>> from transformers import VisionEncoderDecoderModel

        >>> # initialize a vit-bert from a pretrained ViT and a pretrained BERT model. Note that the cross-attention layers will be randomly initialized
        >>> model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        ...     "google/vit-base-patch16-224-in21k", "bert-base-uncased"
        ... )
        >>> # saving model after fine-tuning
        >>> model.save_pretrained("./vit-bert")
        >>> # load fine-tuned model
        >>> model = VisionEncoderDecoderModel.from_pretrained("./vit-bert")
        ```"""

        kwargs_encoder = {
            argument[len("encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                encoder_config, kwargs_encoder = AutoConfig.from_pretrained(
                    encoder_pretrained_model_name_or_path, **kwargs_encoder, return_unused_kwargs=True
                )

                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = AutoModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_decoder:
                decoder_config, kwargs_decoder = AutoConfig.from_pretrained(
                    decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True
                )

                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention"
                        f" layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if"
                        f" {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True

                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False or kwargs_decoder["config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. "
                    f"In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, "
                    "make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` "
                    "passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a "
                    "`decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )

            decoder = AutoModelForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

        # instantiate config with corresponding kwargs
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)

        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        return cls(encoder=encoder, decoder=decoder, config=config)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        label_polys: Optional[torch.LongTensor] = None,
        label_row_spans: Optional[torch.FloatTensor] = None,
        label_col_spans: Optional[torch.FloatTensor] = None,
        label_guiders: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            encoder_outputs = self.encoder(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]

        # optionally project encoder_hidden_states
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        encoder_pos_seq = torch.arange(encoder_hidden_states.shape[1], device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
        encoder_pos_emb = self.encoder_pos_emb(encoder_pos_seq)
        encoder_hidden_states = encoder_hidden_states + encoder_pos_emb[None,]
        
        # else:
        encoder_attention_mask = None

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True, # last hidden state = decoder_outputs[-1][-1]
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # compute pos loss
        pos_loss = None
        if label_polys is not None:
            # pos weight = self.decoder.model.decoder.embed_tokens.weight[-1000:].shape
            dtype = encoder_hidden_states.dtype
            batch_size = decoder_outputs[-1][-1].shape[0]
            pos_key = self.decoder.model.decoder.embed_tokens.weight[-1000:].to(dtype=dtype)[None].repeat(batch_size, 1, 1)
            pos_x1_query = self.poly_x1_proj(decoder_outputs[-1][-1])
            pos_x1_logit = torch.bmm(pos_x1_query, pos_key.transpose(1, 2))
            pos_y1_query = self.poly_y1_proj(decoder_outputs[-1][-1])
            pos_y1_logit = torch.bmm(pos_y1_query, pos_key.transpose(1, 2))
            pos_x2_query = self.poly_x2_proj(decoder_outputs[-1][-1])
            pos_x2_logit = torch.bmm(pos_x2_query, pos_key.transpose(1, 2))
            pos_y2_query = self.poly_y2_proj(decoder_outputs[-1][-1])
            pos_y2_logit = torch.bmm(pos_y2_query, pos_key.transpose(1, 2))
            pos_x3_query = self.poly_x3_proj(decoder_outputs[-1][-1])
            pos_x3_logit = torch.bmm(pos_x3_query, pos_key.transpose(1, 2))
            pos_y3_query = self.poly_y3_proj(decoder_outputs[-1][-1])
            pos_y3_logit = torch.bmm(pos_y3_query, pos_key.transpose(1, 2))
            pos_x4_query = self.poly_x4_proj(decoder_outputs[-1][-1])
            pos_x4_logit = torch.bmm(pos_x4_query, pos_key.transpose(1, 2))
            pos_y4_query = self.poly_y4_proj(decoder_outputs[-1][-1])
            pos_y4_logit = torch.bmm(pos_y4_query, pos_key.transpose(1, 2))

            pos_logit = torch.stack((pos_x1_logit, pos_y1_logit, pos_x2_logit, pos_y2_logit, \
                pos_x3_logit, pos_y3_logit, pos_x4_logit, pos_y4_logit), dim=2)
            
            pos_logit = pos_logit.softmax(-1)
            pos_range = torch.arange(pos_logit.shape[-1], dtype=pos_logit.dtype, device=pos_logit.device)[None, None, None]
            pos_logit = (pos_range * pos_logit).sum(-1) # (B,L,8)

            avail_mask = (label_polys.reshape(-1) != -100).float()
            pos_loss  = F.l1_loss(pos_logit.reshape(-1), label_polys.reshape(-1), reduction='none')
            pos_loss = (pos_loss * avail_mask).sum() / (avail_mask.sum() + 1e-6)

            counter.update(dict(pos_loss=pos_loss))

        # compute row span loss
        row_span_loss = None
        if label_row_spans is not None:
            dtype = encoder_hidden_states.dtype
            batch_size = decoder_outputs[-1][-1].shape[0]
            row_span_key = self.decoder.model.decoder.embed_tokens.weight[-1000:].to(dtype=dtype)[None].repeat(batch_size, 1, 1)
            row_span_query = self.row_span_proj(decoder_outputs[-1][-1])
            row_span_logit = torch.bmm(row_span_query, row_span_key.transpose(1, 2)) # B, SeqLen, PosLen
            row_span_loss = sigmoid_focal_loss(pred=row_span_logit, target=label_row_spans, reduction='none')
            row_span_loss = (row_span_loss * (label_row_spans != -100).float()).sum() / ((label_row_spans > 0).float().sum() + 1e-6) # divide by pos sample

            counter.update(dict(row_span_loss=row_span_loss))
        
        # compute col span loss
        col_span_loss = None
        if label_col_spans is not None:
            dtype = encoder_hidden_states.dtype
            batch_size = decoder_outputs[-1][-1].shape[0]
            col_span_key = self.decoder.model.decoder.embed_tokens.weight[-1000:].to(dtype=dtype)[None].repeat(batch_size, 1, 1)
            col_span_query = self.col_span_proj(decoder_outputs[-1][-1])
            col_span_logit = torch.bmm(col_span_query, col_span_key.transpose(1, 2)) # B, SeqLen, PosLen
            col_span_loss = sigmoid_focal_loss(pred=col_span_logit, target=label_col_spans, reduction='none')
            col_span_loss = (col_span_loss * (label_col_spans != -100).float()).sum() / ((label_col_spans > 0).float().sum() + 1e-6) # divide by pos sample

            counter.update(dict(col_span_loss=col_span_loss))

        # compute guider loss
        guider_loss = None
        if label_guiders is not None:
            guider_query_logit = self.guider_proj(decoder_outputs[-1][-1])
            guider_logit = torch.bmm(guider_query_logit, encoder_hidden_states.transpose(1, 2)) # B, SeqLen, ImgLen
            guider_loss = sigmoid_focal_loss(pred=guider_logit, target=label_guiders, reduction='none')
            guider_loss = (guider_loss * (label_guiders != -100).float()).sum() / ((label_guiders != -100).float().sum() + 1e-6)

            counter.update(dict(guider_loss=guider_loss))

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            prediction_scores = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(prediction_scores.reshape(-1, self.decoder.config.vocab_size), labels.reshape(-1))
            loss = (loss * loss_mask.reshape(-1)).sum() / ((loss_mask != 0).float().sum() + 1e-6)
            
            counter.update(dict(lm_loss=loss))

        if loss is not None and pos_loss is not None and row_span_loss is not None and col_span_loss is not None:
            # loss = loss + pos_loss + row_span_loss + col_span_loss
            loss = self.awl(loss, pos_loss, row_span_loss, col_span_loss, guider_loss)
        
        counter.update(dict(lm_weight=self.awl.params[0], pos_weight=self.awl.params[1], \
            row_span_weight=self.awl.params[2], col_span_weight=self.awl.params[3]))

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        return input_dict

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the VisionEncoderDecoderModel directly is not supported.Please use the"
            " respective methods of the wrapped decoder object (model.decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past_key_values, beam_idx)
