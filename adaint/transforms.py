import random
import numpy as np
from PIL import Image
from torch.nn.modules.utils import _pair
from torchvision.transforms import ColorJitter

from mmedit.datasets.registry import PIPELINES


@PIPELINES.register_module()
class RandomRatioCrop(object):
    """Crops an image by a random ratio.

    This data augmentation technique randomly crops a portion of an image.
    The size of the crop is determined by a random ratio, which can be
    isotropic (same ratio for height and width) or anisotropic.

    Args:
        keys (list[str]): A list of keys corresponding to the images to be
            cropped in the results dictionary.
        crop_ratio (tuple[float]): A range `(min_ratio, max_ratio)` from
            which the crop ratio will be uniformly sampled.
        isotropic (bool, optional): If True, the same crop ratio is used for
            both height and width. Defaults to False.
    """

    def __init__(self, keys, crop_ratio, isotropic=False):
        self.crop_ratio = _pair(crop_ratio)
        self.isotropic = isotropic
        self.keys = keys

    def _get_cropbox(self, img):
        ratio_h = random.uniform(*self.crop_ratio)
        ratio_w = ratio_h if self.isotropic else random.uniform(*self.crop_ratio)
        crop_size = (int(img.shape[0] * ratio_h), int(img.shape[1] * ratio_w))
        margin_h = max(img.shape[0] - crop_size[0], 0)
        margin_w = max(img.shape[1] - crop_size[1], 0)
        offset_h = random.randint(0, margin_h)
        offset_w = random.randint(0, margin_w)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]
        return crop_y1, crop_y2, crop_x1, crop_x2
    
    def __call__(self, results):
        """Applies the random crop to the images in the results dictionary.

        Args:
            results (dict): A dictionary containing the images to be
                cropped.

        Returns:
            dict: The results dictionary with the cropped images.
        """
        y1, y2, x1, x2 = self._get_cropbox(results[self.keys[0]])
        for key in self.keys:
            results[key] = results[key][y1:y2, x1:x2, :]
            results[f'{key}_crop_size'] = results[key].shape[:2]
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(crop_ratio={}, isotropic={}, keys={})'.format(
            self.crop_ratio, self.isotropic, self.keys)
        
        
@PIPELINES.register_module()
class FlexibleRescaleToZeroOne(object):
    """Rescales image pixel values to the range [0, 1].

    This transform is more flexible than the standard `RescaleToZeroOne`
    as it supports both 8-bit and 16-bit integer inputs, converting them
    to a floating-point representation between 0 and 1.

    Args:
        keys (list[str]): A list of keys corresponding to the images to be
            rescaled in the results dictionary.
        precision (int, optional): The desired floating-point precision of
            the output. Can be 16, 32, or 64. Defaults to 32.
    """

    def __init__(self, keys, precision=32):
        assert precision in [16, 32, 64]
        self.precision = 'float{}'.format(precision)
        self.keys = keys

    def _to_float(self, img):
        if img.dtype == np.uint8:
            factor = 255
        elif img.dtype == np.uint16:
            factor = 65535
        else:
            factor = 1
        img = img.astype(self.precision) / factor
        return img.clip(0, 1)

    def __call__(self, results):
        """Applies the rescaling to the images in the results dictionary.

        Args:
            results (dict): A dictionary containing the images to be
                rescaled.

        Returns:
            dict: The results dictionary with the rescaled images.
        """
        for key in self.keys:
            results[key] = self._to_float(results[key])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(precision={}, keys={})'.format(
            self.precision, self.keys)
        return repr_str
    


@PIPELINES.register_module
class RandomColorJitter(object):
    """Randomly jitters the color of an image.

    This transform randomly adjusts the brightness, contrast, saturation, and
    hue of an image. It serves as a wrapper around the `ColorJitter` transform
    from torchvision, making it compatible with the MMEditing data pipeline.

    Args:
        keys (list[str]): A list of keys corresponding to the images to be
            jittered in the results dictionary.
        brightness (float or tuple[float], optional): How much to jitter
            brightness. A value of 0 means no jitter. Defaults to 0.
        contrast (float or tuple[float], optional): How much to jitter
            contrast. A value of 0 means no jitter. Defaults to 0.
        saturation (float or tuple[float], optional): How much to jitter
            saturation. A value of 0 means no jitter. Defaults to 0.
        hue (float or tuple[float], optional): How much to jitter hue.
            A value of 0 means no jitter. Defaults to 0.
    """

    def __init__(self, keys, brightness=0, contrast=0, saturation=0, hue=0):
        self.keys = keys
        self._transform = ColorJitter(brightness, contrast, saturation, hue)

    def transform(self, img):
        return np.array(self._transform(Image.fromarray(img)))

    def __call__(self, results):
        """Applies the color jitter to the images in the results dictionary.

        Args:
            results (dict): A dictionary containing the images to be
                jittered.

        Returns:
            dict: The results dictionary with the jittered images.
        """
        for key in self.keys:
            results[key] = self.transform(results[key])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(brightness={}, contrast={}, saturation={}, hue={}, keys={})'.format(
            self._transform.brightness, self._transform.contrast,
            self._transform.saturation, self._transform.hue, self.keys)
        


@PIPELINES.register_module()
class FlipChannels(object):
    """Flips the color channels of an image (e.g., RGB to BGR).

    This is often used when working with models that expect a different
    channel order than the one provided by the data loading library.

    Args:
        keys (list[str]): A list of keys corresponding to the images whose
            channels will be flipped in the results dictionary.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Applies the channel flip to the images in the results dictionary.

        Args:
            results (dict): A dictionary containing the images whose channels
                will be flipped.

        Returns:
            dict: The results dictionary with the channel-flipped images.
        """
        for key in self.keys:
            results[key] = results[key][..., ::-1]
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(keys={})'.format(self.keys)