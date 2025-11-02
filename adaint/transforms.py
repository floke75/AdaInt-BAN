"""Custom data transforms used by the AdaInt pipelines.

Each transform operates on a dictionary produced by the MMEditing data
pipeline.  The docstrings capture the exact contract expected by the rest of
the codebase so that other agents (and humans!) can confidently reuse these
utilities without having to reverse engineer them from the implementation.
"""

import random
import numpy as np
from PIL import Image
from torch.nn.modules.utils import _pair
from torchvision.transforms import ColorJitter

from mmedit.datasets.registry import PIPELINES


@PIPELINES.register_module()
class RandomRatioCrop(object):
    """Crops an image by a randomly sampled ratio.

    The crop is applied to all images referenced by ``keys``.  The ratio can
    either be a single float – in which case the same lower and upper bound is
    used for both height and width – or an explicit ``(min, max)`` pair.  When
    ``isotropic`` is ``True`` the sampled height ratio is reused for the
    width, otherwise height and width are sampled independently.

    Args:
        keys (list[str]): Keys of the images to crop from ``results``.
        crop_ratio (float or tuple[float, float]): Bounds for the uniformly
            sampled crop ratio.  If a float is provided the bounds become
            ``(crop_ratio, crop_ratio)``.
        isotropic (bool, optional): Whether to use the same ratio for both
            spatial dimensions. Defaults to ``False``.
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
        """Apply the sampled crop box to all requested images.

        Args:
            results (dict): MMEditing style results dictionary containing
                NumPy images in ``HWC`` layout.

        Returns:
            dict: ``results`` with in-place cropped images and the auxiliary
            ``"{key}_crop_size"`` entries describing the resulting ``(H, W)``.
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
    """Normalise image pixel values to the ``[0, 1]`` range.

    MMEditing's default transform only supports 8-bit inputs.  The variant used
    by AdaInt also handles 16-bit integer arrays and already normalised floating
    point inputs.  Values are cast to the requested floating-point precision and
    clipped to ``[0, 1]`` to guard against minor numerical noise.

    Args:
        keys (list[str]): Keys of images to rescale.
        precision (int, optional): Target floating precision (``16``, ``32`` or
            ``64``). Defaults to ``32``.
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
        """Apply normalisation in-place to all requested images."""
        for key in self.keys:
            results[key] = self._to_float(results[key])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(precision={}, keys={})'.format(
            self.precision, self.keys)
        return repr_str
    
@PIPELINES.register_module()
class RandomColorJitter(object):
    """Apply :class:`~torchvision.transforms.ColorJitter` to NumPy images.

    The underlying torchvision transform expects PIL images or tensors.  The
    wrapper converts the ``HWC`` NumPy arrays produced by the MMEditing data
    pipeline to ``PIL.Image`` objects on the fly and converts the result back to
    NumPy.

    Args:
        keys (list[str]): Keys of images to jitter.
        brightness (float or tuple[float], optional): Brightness jitter range.
        contrast (float or tuple[float], optional): Contrast jitter range.
        saturation (float or tuple[float], optional): Saturation jitter range.
        hue (float or tuple[float], optional): Hue jitter range.
    """

    def __init__(self, keys, brightness=0, contrast=0, saturation=0, hue=0):
        self.keys = keys
        self._transform = ColorJitter(brightness, contrast, saturation, hue)

    def transform(self, img):
        return np.array(self._transform(Image.fromarray(img)))

    def __call__(self, results):
        """Apply colour jittering in-place for all configured keys."""
        for key in self.keys:
            results[key] = self.transform(results[key])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(brightness={}, contrast={}, saturation={}, hue={}, keys={})'.format(
            self._transform.brightness, self._transform.contrast,
            self._transform.saturation, self._transform.hue, self.keys)
        return repr_str
        


@PIPELINES.register_module()
class FlipChannels(object):
    """Reverse the order of the last channel dimension of NumPy images.

    Args:
        keys (list[str]): Keys of images whose channel order should be
            reversed.  This is typically used to convert between ``RGB`` and
            ``BGR`` layouts.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Reverse the channel order for every configured image in-place."""
        for key in self.keys:
            results[key] = results[key][..., ::-1]
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(keys={})'.format(self.keys)