import numbers
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import mmcv
from mmcv.runner import auto_fp16

from mmedit.models.base import BaseModel
from mmedit.models.registry import MODELS
from mmedit.models.builder import build_loss
from mmedit.core import psnr, ssim, tensor2img
from mmedit.utils import get_root_logger

from ailut import ailut_transform


class BasicBlock(nn.Sequential):
    """A basic building block for convolutional neural networks.

    This block consists of a 2D convolution layer followed by a LeakyReLU
    activation. Optionally, it can include an instance normalization layer.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int, optional): The size of the convolutional kernel.
            Defaults to 3.
        stride (int, optional): The stride of the convolution. Defaults to 1.
        norm (bool, optional): Whether to include an instance normalization
            layer. Defaults to False.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm=False):
        body = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1),
            nn.LeakyReLU(0.2)
        ]
        if norm:
            body.append(nn.InstanceNorm2d(out_channels, affine=True))
        super(BasicBlock, self).__init__(*body)


class TPAMIBackbone(nn.Sequential):
    """A 5-layer CNN backbone for image feature extraction.

    This backbone is based on the architecture described in the TPAMI 3D-LUT
    paper. It processes an input image, extracts a feature vector, and
    prepares it for the LUT generator.

    Args:
        pretrained (bool, optional): This argument is ignored as no pretrained
            weights are used. Defaults to False.
        input_resolution (int, optional): The resolution to which input images
            are downsampled before feature extraction. Defaults to 256.
        extra_pooling (bool, optional): If True, an extra adaptive average
            pooling layer is added at the end of the backbone. This reduces
            the spatial dimensions of the output feature map, which can
            decrease the number of parameters in subsequent layers.
            Defaults to False.
    """

    def __init__(self, pretrained=False, input_resolution=256, extra_pooling=False):
        body = [
            BasicBlock(3, 16, stride=2, norm=True),
            BasicBlock(16, 32, stride=2, norm=True),
            BasicBlock(32, 64, stride=2, norm=True),
            BasicBlock(64, 128, stride=2, norm=True),
            BasicBlock(128, 128, stride=2),
            nn.Dropout(p=0.5),
        ]
        if extra_pooling:
            body.append(nn.AdaptiveAvgPool2d(2))
        super().__init__(*body)
        self.input_resolution = input_resolution
        self.out_channels = 128 * (4 if extra_pooling else 64)

    def forward(self, imgs):
        """Processes the input image tensor to extract features.

        The input images are first resized to a fixed resolution and then
        passed through the convolutional layers. The resulting feature map
        is flattened into a vector.

        Args:
            imgs (torch.Tensor): A batch of input images with shape
                (b, c, h, w).

        Returns:
            torch.Tensor: A batch of flattened feature vectors with shape
                (b, f), where f is the number of features.
        """
        imgs = F.interpolate(imgs, size=(self.input_resolution,) * 2,
            mode='bilinear', align_corners=False)
        return super().forward(imgs).view(imgs.shape[0], -1)


class Res18Backbone(nn.Module):
    """A ResNet-18 backbone for image feature extraction.

    This module uses a pretrained ResNet-18 model from torchvision to extract
    features from input images. The final fully connected layer is replaced
    with an identity mapping to output the feature vector.

    Args:
        pretrained (bool, optional): If True, loads weights pretrained on
            ImageNet. Defaults to True.
        input_resolution (int, optional): The resolution to which input images
            are downsampled before feature extraction. Defaults to 224.
        extra_pooling (bool, optional): This argument is ignored.
    """

    def __init__(self, pretrained=True, input_resolution=224, extra_pooling=False):
        super().__init__()
        net = torchvision.models.resnet18(pretrained=pretrained)
        net.fc = nn.Identity()
        self.net = net
        self.input_resolution = input_resolution
        self.out_channels = 512

    def forward(self, imgs):
        """Processes the input image tensor to extract features.

        The input images are first resized to a fixed resolution and then
        passed through the ResNet-18 model. The resulting feature map is
        flattened into a vector.

        Args:
            imgs (torch.Tensor): A batch of input images with shape
                (b, c, h, w).

        Returns:
            torch.Tensor: A batch of flattened feature vectors with shape
                (b, f), where f is the number of features.
        """
        imgs = F.interpolate(imgs, size=(self.input_resolution,) * 2,
            mode='bilinear', align_corners=False)
        return self.net(imgs).view(imgs.shape[0], -1)


class LUTGenerator(nn.Module):
    """Generates a 3D Lookup Table (LUT) from an image feature vector.

    This module takes a feature vector extracted from an image and generates
    a corresponding 3D LUT. The LUT is constructed as a linear combination of
    a set of basis LUTs, where the combination weights are learned from the
    input features.

    This corresponds to the mapping `h` in the paper.

    Args:
        n_colors (int): The number of color channels in the input image (e.g.,
            3 for RGB).
        n_vertices (int): The number of sampling points (vertices) along each
            dimension of the 3D LUT.
        n_feats (int): The dimensionality of the input feature vector.
        n_ranks (int): The number of basis LUTs to use for constructing the
            final LUT. This is also referred to as the number of ranks.
    """

    def __init__(self, n_colors, n_vertices, n_feats, n_ranks) -> None:
        super().__init__()

        # h0
        self.weights_generator = nn.Linear(n_feats, n_ranks)
        # h1
        self.basis_luts_bank = nn.Linear(
            n_ranks, n_colors * (n_vertices ** n_colors), bias=False)

        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.n_feats = n_feats
        self.n_ranks = n_ranks

    def init_weights(self):
        r"""Init weights for models.

        For the mapping f (`backbone`) and h (`lut_generator`), we follow the initialization in
            [TPAMI 3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).

        """
        nn.init.ones_(self.weights_generator.bias)
        identity_lut = torch.stack([
            torch.stack(
                torch.meshgrid(*[torch.arange(self.n_vertices) for _ in range(self.n_colors)]),
                dim=0).div(self.n_vertices - 1).flip(0),
            *[torch.zeros(
                self.n_colors, *((self.n_vertices,) * self.n_colors)) for _ in range(self.n_ranks - 1)]
            ], dim=0).view(self.n_ranks, -1)
        self.basis_luts_bank.weight.data.copy_(identity_lut.t())

    def forward(self, x):
        """Generates a batch of 3D LUTs from a batch of feature vectors.

        Args:
            x (torch.Tensor): A batch of input feature vectors with shape
                (b, f), where b is the batch size and f is the number of
                features.

        Returns:
            A tuple containing:
            - torch.Tensor: The learned combination weights for the basis
              LUTs, with shape (b, r), where r is the number of ranks.
            - torch.Tensor: The generated 3D LUTs, with shape
              (b, c, d, d, d), where c is the number of color channels and d
              is the number of vertices.
        """
        weights = self.weights_generator(x)
        luts = self.basis_luts_bank(weights)
        luts = luts.view(x.shape[0], -1, *((self.n_vertices,) * self.n_colors))
        return weights, luts

    def regularizations(self, smoothness, monotonicity):
        """Computes regularization terms for the basis LUTs.

        This method calculates two regularization terms:
        - Smoothness: Encourages the basis LUTs to be smooth, penalizing
          large differences between adjacent vertices.
        - Monotonicity: Encourages the basis LUTs to be monotonically
          increasing, which is a desirable property for image enhancement
          to avoid color artifacts.

        Args:
            smoothness (float): The weight for the smoothness regularization
                term.
            monotonicity (float): The weight for the monotonicity
                regularization term.

        Returns:
            A tuple containing:
            - torch.Tensor: The smoothness regularization loss.
            - torch.Tensor: The monotonicity regularization loss.
        """
        basis_luts = self.basis_luts_bank.weight.t().view(
            self.n_ranks, self.n_colors, *((self.n_vertices,) * self.n_colors))
        tv, mn = 0, 0
        for i in range(2, basis_luts.ndimension()):
            diff = torch.diff(basis_luts.flip(i), dim=i)
            tv += torch.square(diff).sum(0).mean()
            mn += F.relu(diff).sum(0).mean()
        reg_smoothness = smoothness * tv
        reg_monotonicity = monotonicity * mn
        return reg_smoothness, reg_monotonicity


class AdaInt(nn.Module):
    """The Adaptive Interval Learning (AdaInt) module.

    This module learns non-uniform sampling intervals for the 3D LUT,
    allowing for a more adaptive and expressive color transformation. It
    consists of a single fully-connected layer that predicts the sampling
    intervals from an image feature vector.

    This corresponds to the mapping `g` in the paper.

    Args:
        n_colors (int): The number of color channels.
        n_vertices (int): The number of sampling points (vertices) along each
            dimension of the LUT.
        n_feats (int): The dimensionality of the input feature vector.
        adaint_share (bool, optional): If True, the same set of sampling
            intervals is shared across all color channels. This can reduce
            the number of parameters. Defaults to False.
    """

    def __init__(self, n_colors, n_vertices, n_feats, adaint_share=False) -> None:
        super().__init__()
        repeat_factor = n_colors if not adaint_share else 1
        self.intervals_generator = nn.Linear(
            n_feats, (n_vertices - 1) * repeat_factor)

        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.adaint_share = adaint_share

    def init_weights(self):
        r"""Init weights for models.

        We use all-zero and all-one initializations for its weights and bias, respectively.
        """
        nn.init.zeros_(self.intervals_generator.weight)
        nn.init.ones_(self.intervals_generator.bias)

    def forward(self, x):
        """Generates the sampling coordinates for the 3D LUT.

        Args:
            x (torch.Tensor): A batch of input feature vectors with shape
                (b, f), where b is the batch size and f is the number of
                features.

        Returns:
            torch.Tensor: The learned sampling coordinates (vertices) for
                the 3D LUT, with shape (b, c, d), where c is the number of
                color channels and d is the number of vertices.
        """
        x = x.view(x.shape[0], -1)
        intervals = self.intervals_generator(x).view(
            x.shape[0], -1, self.n_vertices - 1)
        if self.adaint_share:
            intervals = intervals.repeat_interleave(self.n_colors, dim=1)
        intervals = intervals.softmax(-1)
        vertices = F.pad(intervals.cumsum(-1), (1, 0), 'constant', 0)
        return vertices


@MODELS.register_module()
class AiLUT(BaseModel):
    """Adaptive-Interval 3D Lookup Table for real-time image enhancement.

    This class implements the core of the AiLUT model. It integrates a
    backbone for feature extraction, a LUT generator, and the AdaInt module
    to create an image-adaptive 3D LUT. This LUT is then used to transform
    the input image for enhancement.

    Args:
        n_ranks (int, optional): The number of basis LUTs. Defaults to 3.
        n_vertices (int, optional): The number of sampling points along each
            dimension of the LUT. Defaults to 33.
        en_adaint (bool, optional): If True, enables the AdaInt module for
            adaptive interval learning. Defaults to True.
        en_adaint_share (bool, optional): If True, shares sampling intervals
            across color channels in AdaInt. Only effective if `en_adaint`
            is True. Defaults to False.
        backbone (str, optional): The architecture of the backbone network.
            Can be either 'tpami' or 'res18'. Defaults to 'tpami'.
        pretrained (bool, optional): If True, loads pretrained weights for the
            backbone (only applicable to 'res18'). Defaults to False.
        n_colors (int, optional): The number of color channels. Defaults to 3.
        sparse_factor (float, optional): The weight for the sparse
            regularization loss on the LUT combination weights. Defaults to
            0.0001.
        smooth_factor (float, optional): The weight for the smoothness
            regularization loss on the basis LUTs. Defaults to 0.
        monotonicity_factor (float, optional): The weight for the
            monotonicity regularization loss on the basis LUTs. Defaults to
            10.0.
        recons_loss (dict, optional): Configuration for the reconstruction
            loss between the enhanced and ground-truth images. Defaults to
            `dict(type='L2Loss', loss_weight=1.0, reduction='mean')`.
        train_cfg (dict, optional): Configuration for training, such as the
            number of iterations to fix the AdaInt module. Defaults to None.
        test_cfg (dict, optional): Configuration for testing, including
            metrics to evaluate. Defaults to None.
    """

    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self,
        n_ranks=3,
        n_vertices=33,
        en_adaint=True,
        en_adaint_share=False,
        backbone='tpami',
        pretrained=False,
        n_colors=3,
        sparse_factor=0.0001,
        smooth_factor=0,
        monotonicity_factor=10.0,
        recons_loss=dict(type='L2Loss', loss_weight=1.0, reduction='mean'),
        train_cfg=None,
        test_cfg=None):

        super().__init__()

        assert backbone.lower() in ['tpami', 'res18']

        # mapping f
        self.backbone = dict(
            tpami=TPAMIBackbone,
            res18=Res18Backbone)[backbone.lower()](pretrained, extra_pooling=en_adaint)

        # mapping h
        self.lut_generator = LUTGenerator(
            n_colors, n_vertices, self.backbone.out_channels, n_ranks)

        # mapping g
        if en_adaint:
            self.adaint = AdaInt(
                n_colors, n_vertices, self.backbone.out_channels, en_adaint_share)
        else:
            uniform_vertices = torch.arange(n_vertices).div(n_vertices - 1) \
                                    .repeat(n_colors, 1)
            self.register_buffer('uniform_vertices', uniform_vertices.unsqueeze(0))

        self.n_ranks = n_ranks
        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.en_adaint = en_adaint
        self.sparse_factor = sparse_factor
        self.smooth_factor = smooth_factor
        self.monotonicity_factor = monotonicity_factor
        self.backbone_name = backbone.lower()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.fp16_enabled = False

        self.init_weights()

        self.recons_loss = build_loss(recons_loss)

        # fix AdaInt for some steps
        self.n_fix_iters = train_cfg.get('n_fix_iters', 0) if train_cfg else 0
        self.adaint_fixed = False
        self.register_buffer('cnt_iters', torch.zeros(1))

    def init_weights(self):
        """Initializes the weights of the model's submodules.

        This method applies a specific initialization strategy to the different
        components of the AiLUT model:
        - The `backbone` is initialized using Xavier normalization for convolutional
          layers and normal distribution for instance normalization layers, unless
          it's a pretrained ResNet-18.
        - The `lut_generator` is initialized to approximate an identity
          transformation at the beginning of training.
        - The `adaint` module is initialized with zeros for its weights and ones
          for its bias to start with uniform sampling intervals.
        """
        def special_initilization(m):
            classname = m.__class__.__name__
            if 'Conv' in classname:
                nn.init.xavier_normal_(m.weight.data)
            elif 'InstanceNorm' in classname:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
        if self.backbone_name not in ['res18']:
            self.apply(special_initilization)
        self.lut_generator.init_weights()
        if self.en_adaint:
            self.adaint.init_weights()

    def forward_dummy(self, imgs):
        """Executes the core forward pass of the AiLUT model.

        This method takes a batch of images and passes them through the
        backbone, LUT generator, and AdaInt module to produce the final
        enhanced images.

        Args:
            imgs (torch.Tensor): A batch of input images with shape
                (b, c, h, w).

        Returns:
            A tuple containing:
            - torch.Tensor: The enhanced output images.
            - torch.Tensor: The learned LUT combination weights.
            - torch.Tensor: The learned sampling coordinates (vertices).
        """
        # E: (b, f)
        codes = self.backbone(imgs)
        # (b, m), T: (b, c, d, d, d)
        weights, luts = self.lut_generator(codes)
        # \hat{P}: (b, c, d)
        if self.en_adaint:
            vertices = self.adaint(codes)
        else:
            vertices = self.uniform_vertices

        outs = ailut_transform(imgs, luts, vertices)

        return outs, weights, vertices

    @auto_fp16(apply_to=('lq', ))
    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        """The main forward function for the AiLUT model.

        This method dispatches to either the training or testing forward
        function based on the `test_mode` flag.

        Args:
            lq (torch.Tensor): A batch of low-quality input images.
            gt (torch.Tensor, optional): A batch of ground-truth images.
                Required during training. Defaults to None.
            test_mode (bool, optional): If True, the model is in testing
                mode. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary of results, including losses during training
                or evaluation results during testing.
        """

        if test_mode:
            return self.forward_test(lq, gt, **kwargs)

        return self.forward_train(lq, gt)

    def forward_train(self, lq, gt):
        """Executes the forward pass during training.

        This method computes the reconstruction loss and regularization
        losses, and prepares the results for the training step.

        Args:
            lq (torch.Tensor): A batch of low-quality input images.
            gt (torch.Tensor): A batch of ground-truth images.

        Returns:
            dict: A dictionary of results containing the computed losses,
                number of samples, and the lq, gt, and output images.
        """
        losses = dict()
        output, weights, vertices = self.forward_dummy(lq)
        losses['loss_recons'] = self.recons_loss(output, gt)
        if self.sparse_factor > 0:
            losses['loss_sparse'] = self.sparse_factor * torch.mean(weights.pow(2))
        reg_smoothness, reg_monotonicity = self.lut_generator.regularizations(
            self.smooth_factor, self.monotonicity_factor)
        if self.smooth_factor > 0:
            losses['loss_smooth'] = reg_smoothness
        if self.monotonicity_factor > 0:
            losses['loss_mono'] = reg_monotonicity
        outputs = dict(
            losses=losses,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu()))
        return outputs

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Executes the forward pass during testing.

        This method generates the enhanced image and, if ground-truth images
        are provided, computes evaluation metrics. It also supports saving
        the output images.

        Args:
            lq (torch.Tensor): A batch of low-quality input images.
            gt (torch.Tensor, optional): A batch of ground-truth images.
                Defaults to None.
            meta (list[dict], optional): A list of metadata for each image,
                used for saving. Defaults to None.
            save_image (bool, optional): If True, saves the enhanced images.
                Defaults to False.
            save_path (str, optional): The directory where saved images will
                be stored. Defaults to None.
            iteration (int, optional): An iteration number to include in the
                saved image filenames. Defaults to None.

        Returns:
            dict: A dictionary of results, including evaluation metrics and
                the lq, gt, and output images.
        """
        output, _, _ = self.forward_dummy(lq)
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            lq_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(lq_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(output), save_path)

        return results

    def train_step(self, data_batch, optimizer):
        """Performs a single training step.

        This method processes a batch of data, computes the loss, performs
        backpropagation, and updates the model parameters. It also handles
        the logic for fixing and unfixing the AdaInt module during the
        initial stages of training.

        Args:
            data_batch (dict): A dictionary containing a batch of training
                data, including 'lq' and 'gt' images.
            optimizer (torch.optim.Optimizer): The optimizer used to update
                the model parameters.

        Returns:
            dict: A dictionary of results from the training step, including
                the loss and log variables.
        """
        # fix AdaInt in the first several epochs
        if self.en_adaint and self.cnt_iters < self.n_fix_iters:
            if not self.adaint_fixed:
                self.adaint_fixed = True
                self.adaint.requires_grad_(False)
                get_root_logger().info(f'Fix AdaInt for {self.n_fix_iters} iters.')
        elif self.en_adaint and self.cnt_iters == self.n_fix_iters:
            self.adaint.requires_grad_(True)
            if self.adaint_fixed:
                self.adaint_fixed = False
                get_root_logger().info(f'Unfix AdaInt after {self.n_fix_iters} iters.')

        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs.update({'log_vars': log_vars})

        self.cnt_iters += 1
        return outputs

    def val_step(self, data_batch, **kwargs):
        """Performs a single validation step.

        This method processes a batch of validation data and returns the
        output of the model.

        Args:
            data_batch (dict): A dictionary containing a batch of validation
                data.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: The output of the model for the given validation batch.
        """
        output = self.forward_test(**data_batch, **kwargs)
        return output

    def evaluate(self, output, gt):
        """Calculates evaluation metrics for the model's output.

        This method computes metrics such as PSNR and SSIM between the
        model's output and the ground-truth images.

        Args:
            output (torch.Tensor): The output images from the model.
            gt (torch.Tensor): The ground-truth images.

        Returns:
            dict: A dictionary of evaluation results, where the keys are
                the metric names and the values are the computed scores.
        """
        crop_border = self.test_cfg.crop_border

        output = tensor2img(output)
        gt = tensor2img(gt)

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            eval_result[metric] = self.allowed_metrics[metric](
                output, gt, crop_border)
        return eval_result
