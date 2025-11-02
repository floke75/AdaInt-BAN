import os.path as osp
from collections import defaultdict

from mmedit.datasets.registry import DATASETS
from mmedit.datasets.base_dataset import BaseDataset


class BaseEnhanceDataset(BaseDataset):
    """A base dataset for image enhancement tasks.

    This dataset class is designed for image enhancement tasks where each
    low-quality (LQ) image has a corresponding ground-truth (GT) high-quality
    image. It reads image paths from an annotation file and organizes them
    into pairs.

    The dataset assumes that LQ and GT images are stored in separate
    directories and that the filenames in the annotation file are relative
    to these directories.

    Args:
        dir_lq (str): The directory containing the low-quality images.
        dir_gt (str): The directory containing the ground-truth images.
        ann_file (str): The path to the annotation file. This file should
            contain a list of image basenames, one per line.
        pipeline (list[dict]): A list of data preprocessing steps to be
            applied to each image pair.
        test_mode (bool, optional): If True, the dataset is in testing mode.
            Defaults to False.
        filetmpl_lq (str, optional): A template for formatting the LQ image
            filenames. Defaults to '{}.jpg'.
        filetmpl_gt (str, optional): A template for formatting the GT image
            filenames. Defaults to '{}.jpg'.
    """
    
    def __init__(self,
                 dir_lq,
                 dir_gt,
                 ann_file,
                 pipeline,
                 test_mode=False,
                 filetmpl_lq='{}.jpg',
                 filetmpl_gt='{}.jpg'):
        super().__init__(pipeline, test_mode=test_mode)
        
        if not osp.isfile(ann_file):
            raise ValueError('"ann_file" must be a path to annotation txt file.')
        
        self.dir_lq = dir_lq
        self.dir_gt = dir_gt
        self.ann_file = ann_file
        self.filetmpl_lq = filetmpl_lq
        self.filetmpl_gt = filetmpl_gt
        self.data_infos = self.load_annotations()
        
    def load_annotations(self):
        """Loads image paths from the annotation file.

        This method reads the annotation file line by line, where each line
        is expected to be a basename for an image. It then constructs the
        full paths to the low-quality (LQ) and ground-truth (GT) images
        based on the provided directory paths and filename templates.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary
                contains the paths to an LQ and a GT image pair.
        """
        data_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                basename = line.split('\n')[0]
                lq_name = self.filetmpl_lq.format(basename)
                gt_name = self.filetmpl_gt.format(basename)
                data_infos.append(
                    dict(
                        lq_path=osp.join(self.dir_lq, lq_name),
                        gt_path=osp.join(self.dir_gt, gt_name)))
        return data_infos
    
    def evaluate(self, results, logger=None):
        """Evaluates the model's performance on the dataset.

        This method takes a list of evaluation results from the model,
        aggregates them, and computes the average score for each metric.

        Args:
            results (list[dict]): A list of dictionaries, where each
                dictionary contains the evaluation results for a single
                sample in the dataset.
            logger (logging.Logger, optional): A logger for printing
                evaluation results. Defaults to None.

        Returns:
            dict: A dictionary containing the average score for each
                evaluation metric.
        """
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        results = [res['eval_result'] for res in results]  # a list of dict
        eval_result = defaultdict(list)  # a dict of list

        for res in results:
            for metric, val in res.items():
                eval_result[metric].append(val)
        for metric, val_list in eval_result.items():
            assert len(val_list) == len(self), (
                f'Length of evaluation result of {metric} is {len(val_list)}, '
                f'should be {len(self)}')

        # average the results
        eval_result = {
            metric: sum(values) / len(self)
            for metric, values in eval_result.items()
        }

        return eval_result
    

@DATASETS.register_module()
class FiveK(BaseEnhanceDataset):
    """The MIT-Adobe FiveK dataset for image enhancement.

    This class is a simple extension of `BaseEnhanceDataset` and is used to
    load data from the MIT-Adobe FiveK dataset. It does not introduce any
    new functionality but is registered in the MMEditing dataset registry
    to be easily configurable.
    """
    pass


@DATASETS.register_module()
class PPR10K(BaseEnhanceDataset):
    """The PPR10K dataset for image enhancement.

    This class is an extension of `BaseEnhanceDataset` specifically for the
    PPR10K dataset. It overrides the `load_annotations` method to handle
    the specific filename conventions of this dataset, where multiple
    low-quality (LQ) images may correspond to a single ground-truth (GT)
    image.
    """
    
    def load_annotations(self):
        """Loads image paths from the annotation file for the PPR10K dataset.

        This method overrides the base implementation to handle the PPR10K
        dataset's naming convention, where multiple low-quality (LQ) images
        can be associated with a single ground-truth (GT) image. The GT
        filename is derived from the LQ filename by taking the first two
        parts of the basename.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary
                contains the paths to an LQ and a GT image pair.
        """
        data_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                basename = line.split('\n')[0]
                lq_name = self.filetmpl_lq.format(basename)
                gt_name = self.filetmpl_gt.format('_'.join(basename.split('_')[:2]))
                data_infos.append(
                    dict(
                        lq_path=osp.join(self.dir_lq, lq_name),
                        gt_path=osp.join(self.dir_gt, gt_name)))
        return data_infos
    