#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THCAtomics.cuh>

constexpr uint32_t AT_APPLY_THREADS_PER_BLOCK = 512;

template <typename T>
__host__ __device__ __forceinline__ T ATenCeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

template <int step = 1>
inline bool getApplyGrid(uint64_t totalElements, dim3& grid, int64_t curDevice, int max_threads_per_block=AT_APPLY_THREADS_PER_BLOCK) {
  if (curDevice == -1) return false;
  uint64_t numel_per_thread = static_cast<uint64_t>(max_threads_per_block) * static_cast<uint64_t>(step);
  uint64_t numBlocks = ATenCeilDiv(totalElements, numel_per_thread);
  uint64_t maxGridX = at::cuda::getDeviceProperties(curDevice)->maxGridSize[0];
  if (numBlocks > maxGridX)
    numBlocks = maxGridX;
  grid = dim3(numBlocks);
  return true;
}

constexpr int getApplyBlockSize() {
  return AT_APPLY_THREADS_PER_BLOCK;
}

inline dim3 getApplyBlock(int max_threads_per_block=AT_APPLY_THREADS_PER_BLOCK) {
  return dim3(max_threads_per_block);
}

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(getApplyBlockSize())
__global__ void roll_cuda_kernel(
    scalar_t* in,
    scalar_t* out,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t shift,
    int64_t size,
    int64_t stride) {
  int64_t linear_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear_index >= N) {
    return;
  }
  int64_t c_index = (linear_index / W / H) % C;
  int64_t start = (size - shift * c_index) % size;
  if (start < 0) start = start + size;
  // roll dim idx is the index of linear_index along the rolling dimension.
  int64_t roll_dim_idx = linear_index % (stride * size) / stride;
  // index into the source data to find appropriate value.
  int64_t source_idx = 0;
  if( roll_dim_idx >= (size - start) ) {
    source_idx = linear_index - ((size - start) * stride);
  } else {
    source_idx = linear_index + (start * stride);
  }
  out[linear_index] = in[source_idx];
}

namespace hsi {

at::Tensor roll_optim_cuda(const at::Tensor& in,
                                   const int shift,
                                   const int dim) {
  AT_ASSERTM(in.device().is_cuda(), "dw must be a CUDA tensor");

  auto out = at::zeros_like(in);
  if (out.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return out;
  }

  const int64_t N = in.numel();
  const int64_t C = in.size(1);
  const int64_t H = in.size(2);
  const int64_t W = in.size(3);
  const int64_t size = in.size(dim);

  dim3 dim_block = getApplyBlock();
  dim3 dim_grid;
  TORCH_CHECK(getApplyGrid(N, dim_grid, in.get_device()), "unable to get dim grid");

  AT_DISPATCH_FLOATING_TYPES(in.scalar_type(), "roll_optim", [&] {
    roll_cuda_kernel<scalar_t><<<dim_grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
      in.contiguous().data_ptr<scalar_t>(),
      out.contiguous().data_ptr<scalar_t>(),
      N, C, H, W, shift, size, in.stride(dim));
  });

  AT_CUDA_CHECK(cudaGetLastError());
  return out;
}

}  // namespace hsi
