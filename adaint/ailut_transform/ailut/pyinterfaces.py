from typing import Tuple

import torch
from torch.cuda.amp import custom_fwd, custom_bwd

from ._ext import (
    lut_cforward, lut_cbackward,
    ailut_cforward, ailut_cbackward
)


class LUTTransformFunction(torch.autograd.Function):
    """A custom autograd function for the standard 3D LUT transform.

    This class defines the forward and backward passes for a standard 3D
    Lookup Table transformation. It is designed to be called from PyTorch
    modules and supports automatic differentiation.
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx,
                img: torch.Tensor,
                lut: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of the 3D LUT transformation.

        Args:
            ctx: The context object for saving information for the backward
                pass.
            img (torch.Tensor): The input image tensor.
            lut (torch.Tensor): The 3D Lookup Table.

        Returns:
            torch.Tensor: The transformed output image.
        """
        img = img.contiguous()
        lut = lut.contiguous()

        assert img.ndimension() == 4, \
            "only support 2D image with batch and channel dimensions (4D tensor)"
        assert lut.ndimension() in [5], \
            "only support 3D lookup table with batch dimension (5D tensor)"

        output = img.new_zeros((img.size(0), lut.size(1), img.size(2), img.size(3)))
        lut_cforward(img, lut, output)

        ctx.save_for_backward(img, lut)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:
        """Performs the backward pass of the 3D LUT transformation.

        This method computes the gradients of the loss with respect to the
        input image and the 3D LUT.

        Args:
            ctx: The context object containing saved tensors from the
                forward pass.
            grad_output (torch.Tensor): The gradient of the loss with
                respect to the output of the forward pass.

        Returns:
            A tuple containing the gradients for the input image and the
            3D LUT.
        """
        grad_output = grad_output.contiguous()

        img, lut = ctx.saved_tensors

        grad_img = torch.zeros_like(img)
        grad_lut = torch.zeros_like(lut)

        lut_cbackward(grad_output, img, lut, grad_img, grad_lut)

        return grad_img, grad_lut


class AiLUTTransformFunction(torch.autograd.Function):
    """A custom autograd function for the Adaptive Interval LUT transform.

    This class defines the forward and backward passes for the AiLUT
    transformation, which uses non-uniform sampling intervals. It is
    designed to be called from PyTorch modules and supports automatic
    differentiation for the image, the LUT, and the sampling vertices.
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx,
                img: torch.Tensor,
                lut: torch.Tensor,
                vertices: torch.tensor) -> torch.Tensor:
        """Performs the forward pass of the AiLUT transformation.

        Args:
            ctx: The context object for saving information for the backward
                pass.
            img (torch.Tensor): The input image tensor.
            lut (torch.Tensor): The 3D Lookup Table.
            vertices (torch.Tensor): The sampling coordinates (vertices) for
                the LUT.

        Returns:
            torch.Tensor: The transformed output image.
        """
        img = img.contiguous()
        lut = lut.contiguous()
        vertices = vertices.contiguous()

        assert img.ndimension() == 4, \
            "only support 2D image with batch and channel dimensions (4D tensor)"
        assert lut.ndimension() in [5], \
            "only support 3D lookup table with batch dimension (5D tensor)"
        assert vertices.ndimension() == 3, \
            "only support 1D vertices list with batch and channel dimensions (3D tensor)"

        output = img.new_zeros((img.size(0), lut.size(1), img.size(2), img.size(3)))
        ailut_cforward(img, lut, vertices, output)

        ctx.save_for_backward(img, lut, vertices)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:
        """Performs the backward pass of the AiLUT transformation.

        This method computes the gradients of the loss with respect to the
        input image, the 3D LUT, and the sampling vertices.

        Args:
            ctx: The context object containing saved tensors from the
                forward pass.
            grad_output (torch.Tensor): The gradient of the loss with
                respect to the output of the forward pass.

        Returns:
            A tuple containing the gradients for the input image, the 3D
            LUT, and the sampling vertices.
        """
        grad_output = grad_output.contiguous()

        img, lut, vertices = ctx.saved_tensors

        grad_img = torch.zeros_like(img)
        grad_lut = torch.zeros_like(lut)
        grad_ver = torch.zeros_like(vertices)

        ailut_cbackward(grad_output, img, lut, vertices,
            grad_img, grad_lut, grad_ver)

        return grad_img, grad_lut, grad_ver


def ailut_transform(
    img: torch.Tensor,
    lut: torch.Tensor,
    vertices: torch.Tensor) -> torch.Tensor:
    """Applies the Adaptive Interval 3D Lookup Table Transform.

    This function serves as a user-friendly interface to the
    `AiLUTTransformFunction`, applying a 3D LUT with non-uniform sampling
    intervals to an image.

    Args:
        img (torch.Tensor): The input image of shape (b, 3, h, w).
        lut (torch.Tensor): The output values of the 3D LUT, with shape
            (b, 3, d, d, d).
        vertices (torch.Tensor): The sampling coordinates for each dimension
            of the 3D LUT, with shape (b, 3, d).

    Returns:
        torch.Tensor: The transformed image of shape (b, 3, h, w).
    """
    return AiLUTTransformFunction.apply(img, lut, vertices)


def lut_transform(
    img: torch.Tensor,
    lut: torch.Tensor) -> torch.Tensor:
    """Applies the Standard 3D Lookup Table Transform.

    This function serves as a user-friendly interface to the
    `LUTTransformFunction`, applying a standard 3D LUT with uniform sampling
    intervals to an image.

    Args:
        img (torch.Tensor): The input image of shape (b, 3, h, w).
        lut (torch.Tensor): The output values of the 3D LUT, with shape
            (b, 3, d, d, d).

    Returns:
        torch.Tensor: The transformed image of shape (b, 3, h, w).
    """
    return LUTTransformFunction.apply(img, lut)
