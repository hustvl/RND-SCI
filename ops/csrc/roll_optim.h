#pragma once
#include <torch/extension.h>
#include <vector>

namespace hsi {

at::Tensor roll_optim_cuda(
  const at::Tensor& in,
  const int shift,
  const int dim);

at::Tensor roll_optim(const at::Tensor& in,
                              const int shift,
                              const int dim) {
  if (in.device().is_cuda()) {
#ifdef WITH_CUDA
    return roll_optim_cuda(in, shift, dim);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

}  // namespace hsi
