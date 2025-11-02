# Self-Distilled Learning of Adaptive Interval 3D Lookup Tables for Real-Time Image Enhancement

This repository contains the official PyTorch implementation for the paper "Self-Distilled Learning of Adaptive Interval 3D Lookup Tables for Real-Time Image Enhancement". This work introduces a novel approach to image enhancement using 3D Lookup Tables (LUTs) with adaptive interval learning and self-distillation.

## Abstract

The 3D Lookup Table (3D LUT) is an efficient tool for real-time image enhancement. However, existing approaches often suffer from limited adaptability and poor generalizability due to uniform sampling strategies and small dataset sizes. To address these limitations, we propose two key innovations:

1.  **Adaptive Intervals Learning (AdaInt):** A novel module that learns non-uniform sampling intervals within the 3D color space. This allows the model to allocate more sampling points to color ranges that require highly non-linear transformations, thereby increasing the expressiveness of the 3D LUT.
2.  **Born-Again Training:** A self-distillation strategy that enhances the model's generalization ability without increasing inference cost.

Our method, which includes the differentiable **Adaptive Interval LUT Transform (AiLUT-Transform)** operator, achieves state-of-the-art performance on public benchmarks with minimal overhead.

## Model Architecture

The core of our model consists of three main components:

1.  **Backbone (Mapping `f`):** A feature extractor (e.g., ResNet-18 or a custom 5-layer CNN) that takes a downsampled version of the input image and produces a feature vector.
2.  **AdaInt Module (Mapping `g`):** A small neural network that takes the feature vector from the backbone and predicts the non-uniform sampling intervals (vertices) for the 3D LUT.
3.  **LUT Generator (Mapping `h`):** Another neural network that also takes the feature vector and generates the output color values for the 3D LUT.

The predicted vertices and output values are then combined to form an image-adaptive 3D LUT, which is used to transform the original image using the AiLUT-Transform operator.

## Requirements

- Python 3.7+
- PyTorch 1.6+
- CUDA 10.1+
- MMEditing

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/zrk2/AdaInt-BAN.git
    cd AdaInt-BAN
    ```

2.  **Install the custom AiLUT-Transform operator:**

    The AiLUT-Transform operator is implemented as a custom PyTorch extension with CUDA and C++ code. To build and install it, run the following command from the `adaint/ailut_transform` directory:

    ```bash
    cd adaint/ailut_transform
    python setup.py develop
    cd ../..
    ```

## Usage

### Training

To train the AiLUT model, use the `tools/train.py` script with a configuration file. For example, to train on the FiveK dataset:

```bash
python tools/train.py configs/adaint/adaint_fivek.py --work-dir ./work_dirs/adaint_fivek
```

Make sure to update the `data_root` path in the configuration file to point to your dataset location.

### Testing

To evaluate a trained model, use the `tools/test.py` script. You will need to provide the configuration file and the path to the trained model checkpoint.

```bash
python tools/test.py configs/adaint/adaint_fivek.py /path/to/your/checkpoint.pth --save-path ./results/
```

The `--save-path` argument is optional and specifies a directory where the enhanced images will be saved.

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@article{zeng2023self,
  title={Self-Distilled Learning of Adaptive Interval 3D Lookup Tables on Real-Time Image Enhancement},
  author={Zeng, Hui and Zhang, Ruikang and Li, Young-Jung and Kim, Jin-Hwan and Choe, Min-Gyu and Lee, Sung-Jea},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  publisher={IEEE}
}
```
