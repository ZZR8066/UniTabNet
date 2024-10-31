import cv2
import numpy.random as random
from transformers.models.donut.image_processing_donut import *


class PhotoMetricDistortion:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """
    
    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, img):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """
        img = img.astype(np.float32)
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta, self.brightness_delta)
            img += delta
        
        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                img *= alpha
        
        # convert color from BGR to HSV
        # img = mmcv.bgr2hsv(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower, self.saturation_upper)
        
        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360
        
        # convert color from HSV to BGR
        # img = mmcv.hsv2bgr(img)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        
        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                img *= alpha
        
        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]
        
        return img


class ImageProcessor(DonutImageProcessor):
    def pad_image(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        random_padding: bool = True,
        data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Pad the image to the specified size.

        Args:
            image (`np.ndarray`):
                The image to be padded.
            size (`Dict[str, int]`):
                The size `{"height": h, "width": w}` to pad the image to.
            random_padding (`bool`, *optional*, defaults to `False`):
                Whether to use random padding or not.
            data_format (`str` or `ChannelDimension`, *optional*):
                The data format of the output image. If unset, the same format as the input image is used.
        """
        output_height, output_width = size["height"], size["width"]
        input_height, input_width = get_image_size(image)

        delta_width = output_width - input_width
        delta_height = output_height - input_height

        if random_padding:
            pad_top = np.random.randint(low=0, high=delta_height + 1)
            pad_left = np.random.randint(low=0, high=delta_width + 1)
        else:
            pad_top = delta_height // 2
            pad_left = delta_width // 2

        pad_bottom = delta_height - pad_top
        pad_right = delta_width - pad_left

        padding = ((pad_top, pad_bottom), (pad_left, pad_right))
        padded_image = pad(image, padding, data_format=data_format)
        return padded_image, [pad_left, pad_top]

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_thumbnail: bool = None,
        do_align_long_axis: bool = None,
        do_pad: bool = None,
        random_padding: bool = False,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        **kwargs,
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. Shortest edge of the image is resized to min(size["height"],
                size["width"]) with the longest edge resized to keep the input aspect ratio.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_thumbnail (`bool`, *optional*, defaults to `self.do_thumbnail`):
                Whether to resize the image using thumbnail method.
            do_align_long_axis (`bool`, *optional*, defaults to `self.do_align_long_axis`):
                Whether to align the long axis of the image with the long axis of `size` by rotating by 90 degrees.
            do_pad (`bool`, *optional*, defaults to `self.do_pad`):
                Whether to pad the image. If `random_padding` is set to `True`, each image is padded with a random
                amont of padding on each size, up to the largest image size in the batch. Otherwise, all images are
                padded to the largest image size in the batch.
            random_padding (`bool`, *optional*, defaults to `self.random_padding`):
                Whether to use random padding when padding the image. If `True`, each image in the batch with be padded
                with a random amount of padding on each side up to the size of the largest image in the batch.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image pixel values.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: defaults to the channel dimension format of the input image.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        if isinstance(size, (tuple, list)):
            # Previous feature extractor had size in (width, height) format
            size = size[::-1]
        size = get_size_dict(size)
        resample = resample if resample is not None else self.resample
        do_thumbnail = do_thumbnail if do_thumbnail is not None else self.do_thumbnail
        do_align_long_axis = do_align_long_axis if do_align_long_axis is not None else self.do_align_long_axis
        do_pad = do_pad if do_pad is not None else self.do_pad
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if do_resize and size is None:
            raise ValueError("Size must be specified if do_resize is True.")

        if do_rescale and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        if do_pad and size is None:
            raise ValueError("Size must be specified if do_pad is True.")

        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError("Image mean and std must be specified if do_normalize is True.")

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_align_long_axis:
            images = [self.align_long_axis(image, size=size) for image in images]

        if do_resize:
            images = [self.resize(image=image, size=size, resample=resample) for image in images]

        if do_thumbnail:
            images = [self.thumbnail(image=image, size=size) for image in images]

        if do_pad:
            resized_shape = [image.shape[:2] for image in images]
            images = [self.pad_image(image=image, size=size, random_padding=random_padding)for image in images]
            padding = [image[1] for image in images]
            images = [image[0] for image in images]

        if do_rescale:
            images = [self.rescale(image=image, scale=rescale_factor) for image in images]

        if do_normalize:
            images = [self.normalize(image=image, mean=image_mean, std=image_std) for image in images]

        images = [to_channel_dimension_format(image, data_format) for image in images]

        data = {"pixel_values": images, 'resized_shape':resized_shape, "padding": padding}
        return BatchFeature(data=data, tensor_type=return_tensors)