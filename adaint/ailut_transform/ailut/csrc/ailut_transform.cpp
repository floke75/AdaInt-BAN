#include <torch/extension.h>

// ----------------------------------------------------------------------------
// CUDA Forward Declarations
// ----------------------------------------------------------------------------
//
// These declarations announce the existence of CUDA kernel launchers that will
// be defined in a separate .cu file. This allows the C++ code to call the
// CUDA kernels without needing to include the CUDA-specific headers.
//

// Launches the CUDA kernel for the forward pass of the standard 3D LUT transform.
void LutTransformForwardCUDAKernelLauncher(
    const torch::Tensor &input, const torch::Tensor &lut, torch::Tensor output);


// Launches the CUDA kernel for the backward pass of the standard 3D LUT transform.
void LutTransformBackwardCUDAKernelLauncher(
    const torch::Tensor &grad_output, const torch::Tensor &input,
    const torch::Tensor &lut, torch::Tensor grad_inp, torch::Tensor grad_lut);


// Launches the CUDA kernel for the forward pass of the Adaptive Interval LUT transform.
void AiLutTransformForwardCUDAKernelLauncher(
    const torch::Tensor &input, const torch::Tensor &lut,
    const torch::Tensor &vertices, torch::Tensor output);


// Launches the CUDA kernel for the backward pass of the Adaptive Interval LUT transform.
void AiLutTransformBackwardCUDAKernelLauncher(
    const torch::Tensor &grad_output, const torch::Tensor &input,
    const torch::Tensor &lut, const torch::Tensor &vertices,
    torch::Tensor grad_inp, torch::Tensor grad_lut, torch::Tensor grad_ver);


// ----------------------------------------------------------------------------
// CUDA Interface
// ----------------------------------------------------------------------------
//
// These functions serve as a bridge between the main C++ interface and the
// CUDA kernel launchers. They simply call the corresponding kernel launcher.
//

void lut_transform_cuda_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor output) {

    LutTransformForwardCUDAKernelLauncher(input, lut, output);
}


void lut_transform_cuda_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor grad_inp,
    torch::Tensor grad_lut) {

    LutTransformBackwardCUDAKernelLauncher(
        grad_output, input, lut, grad_inp, grad_lut);
}


void ailut_transform_cuda_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    const torch::Tensor &vertices,
    torch::Tensor output) {

    AiLutTransformForwardCUDAKernelLauncher(input, lut, vertices, output);
}


void ailut_transform_cuda_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &lut,
    const torch::Tensor &vertices,
    torch::Tensor grad_inp,
    torch::Tensor grad_lut,
    torch::Tensor grad_ver) {

    AiLutTransformBackwardCUDAKernelLauncher(
        grad_output, input, lut, vertices, grad_inp, grad_lut, grad_ver);
}


// ----------------------------------------------------------------------------
// CPU Forward Declarations
// ----------------------------------------------------------------------------
//
// These declarations announce the existence of CPU-based implementations for
// the LUT transforms. These will be defined in a separate .cpp file.
//

// Declares the CPU implementation for the forward pass of the standard 3D LUT transform.
void lut_transform_cpu_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor output);


// Declares the CPU implementation for the backward pass of the standard 3D LUT transform.
void lut_transform_cpu_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor grad_inp,
    torch::Tensor grad_lut);


// Declares the CPU implementation for the forward pass of the Adaptive Interval LUT transform.
void ailut_transform_cpu_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    const torch::Tensor &vertices,
    torch::Tensor output);


// Declares the CPU implementation for the backward pass of the Adaptive Interval LUT transform.
void ailut_transform_cpu_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &lut,
    const torch::Tensor &vertices,
    torch::Tensor grad_inp,
    torch::Tensor grad_lut,
    torch::Tensor grad_ver);


// ----------------------------------------------------------------------------
// C++ Interface
// ----------------------------------------------------------------------------
//
// These functions are the main entry points for the custom operator. They
// check the input tensors and dispatch to either the CUDA or CPU
// implementation based on the device of the input tensors.
//

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void lut_transform_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor output) {

    if (input.device().is_cuda()) {
        CHECK_INPUT(input);
        CHECK_INPUT(lut);
        CHECK_INPUT(output);

        lut_transform_cuda_forward(input, lut, output);
    } else {
        CHECK_CONTIGUOUS(input);
        CHECK_CONTIGUOUS(lut);
        CHECK_CONTIGUOUS(output);

        lut_transform_cpu_forward(input, lut, output);
    }
}


void lut_transform_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor grad_inp,
    torch::Tensor grad_lut) {

    if (input.device().is_cuda()) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(input);
        CHECK_INPUT(lut);
        CHECK_INPUT(grad_inp);
        CHECK_INPUT(grad_lut);

        lut_transform_cuda_backward(grad_output, input, lut, grad_inp, grad_lut);
    } else {
        CHECK_CONTIGUOUS(grad_output);
        CHECK_CONTIGUOUS(input);
        CHECK_CONTIGUOUS(lut);
        CHECK_CONTIGUOUS(grad_inp);
        CHECK_CONTIGUOUS(grad_lut);

        lut_transform_cpu_backward(grad_output, input, lut, grad_inp, grad_lut);
    }
}


void ailut_transform_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    const torch::Tensor &vertices,
    torch::Tensor output) {

    if (input.device().is_cuda()) {
        CHECK_INPUT(input);
        CHECK_INPUT(lut);
        CHECK_INPUT(vertices);
        CHECK_INPUT(output);

        ailut_transform_cuda_forward(input, lut, vertices, output);
    } else {
        CHECK_CONTIGUOUS(input);
        CHECK_CONTIGUOUS(lut);
        CHECK_CONTIGUOUS(vertices);
        CHECK_CONTIGUOUS(output);

        ailut_transform_cpu_forward(input, lut, vertices, output);
    }
}


void ailut_transform_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &lut,
    const torch::Tensor &vertices,
    torch::Tensor grad_inp,
    torch::Tensor grad_lut,
    torch::Tensor grad_ver) {

    if (input.device().is_cuda()) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(input);
        CHECK_INPUT(lut);
        CHECK_INPUT(vertices);
        CHECK_INPUT(grad_inp);
        CHECK_INPUT(grad_lut);
        CHECK_INPUT(grad_ver);

        ailut_transform_cuda_backward(grad_output, input, lut, vertices, grad_inp, grad_lut, grad_ver);
    } else {
        CHECK_CONTIGUOUS(grad_output);
        CHECK_CONTIGUOUS(input);
        CHECK_CONTIGUOUS(lut);
        CHECK_CONTIGUOUS(vertices);
        CHECK_CONTIGUOUS(grad_inp);
        CHECK_CONTIGUOUS(grad_lut);
        CHECK_CONTIGUOUS(grad_ver);

        ailut_transform_cpu_backward(grad_output, input, lut, vertices, grad_inp, grad_lut, grad_ver);
    }
}


// ----------------------------------------------------------------------------
// Pybind11 Interface
// ----------------------------------------------------------------------------
//
// This section uses pybind11 to create a Python module that exposes the C++
// interface functions to Python. This allows the custom operator to be called
// from PyTorch.
//

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lut_cforward", &lut_transform_forward, "Lut-Transform forward");
  m.def("lut_cbackward", &lut_transform_backward, "Lut-Transform backward");
  m.def("ailut_cforward", &ailut_transform_forward, "AiLut-Transform forward");
  m.def("ailut_cbackward", &ailut_transform_backward, "AiLut-Transform backward");
}

