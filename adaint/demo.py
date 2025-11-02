"""Command-line demo for running AdaInt inference on a single image."""

import argparse
import os

import mmcv
import torch
from mmcv.parallel import collate, scatter

from mmedit.apis import init_model
from mmedit.core import tensor2img
from mmedit.datasets.pipelines import Compose


def enhancement_inference(model, img):
    r"""Run the configured model on ``img`` and return the enhanced tensor.

    The helper mirrors :func:`mmedit.apis.inference_model` but strips the
    ground-truth specific steps from the pipeline so that arbitrary standalone
    images can be processed.

    Args:
        model (torch.nn.Module): Loaded MMEditing model.
        img (str): Path to the input image on disk.

    Returns:
        torch.Tensor: The enhanced image tensor produced by the model.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # remove gt from test_pipeline
    keys_to_remove = ['gt', 'gt_path']
    for key in keys_to_remove:
        for pipeline in list(cfg.test_pipeline):
            if 'key' in pipeline and key == pipeline['key']:
                cfg.test_pipeline.remove(pipeline)
            if 'keys' in pipeline and key in pipeline['keys']:
                pipeline['keys'].remove(key)
                if len(pipeline['keys']) == 0:
                    cfg.test_pipeline.remove(pipeline)
            if 'meta_keys' in pipeline and key in pipeline['meta_keys']:
                pipeline['meta_keys'].remove(key)
    # build the data pipeline
    test_pipeline = Compose(cfg.test_pipeline)
    # prepare data
    data = dict(lq_path=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    # forward the model
    with torch.no_grad():
        result = model(test_mode=True, **data)

    return result['output']


def parse_args():
    """Build and parse the CLI arguments used by :func:`main`."""

    parser = argparse.ArgumentParser(description='Enhancement demo')
    parser.add_argument('config', help='Path to the test config file')
    parser.add_argument('checkpoint', help='Checkpoint file to load')
    parser.add_argument('img_path', help='Path to the input image file')
    parser.add_argument('save_path', help='Destination for the enhanced image')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def main():
    """Entry point for the CLI demo script."""

    args = parse_args()

    if not os.path.isfile(args.img_path):
        raise ValueError('It seems that you did not input a valid "image_path".')

    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    output = enhancement_inference(model, args.img_path)
    output = tensor2img(output)

    mmcv.imwrite(output, args.save_path)

if __name__ == '__main__':
    main()