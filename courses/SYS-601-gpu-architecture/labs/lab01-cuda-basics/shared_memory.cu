// Parallel reduction: sum of N floats using shared-memory tree reduction per block.
// Build: nvcc -std=c++17 -O2 shared_memory.cu -o shared_memory

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

constexpr int BLOCK = 256;
constexpr int SHMEM = BLOCK;

__global__ void reduce_sum_kernel(const float* __restrict__ input, float* __restrict__ partial_out,
                                    int n) {
  __shared__ float sdata[SHMEM];

  const int tid = threadIdx.x;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = (i < n) ? input[i] : 0.f;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial_out[blockIdx.x] = sdata[0];
  }
}

static void check(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
    std::exit(1);
  }
}

int main() {
  const int n = 1 << 20;
  float* h_in = static_cast<float*>(std::malloc(static_cast<size_t>(n) * sizeof(float)));
  double cpu_sum = 0.0;
  for (int i = 0; i < n; ++i) {
    h_in[i] = 1.0f;
    cpu_sum += h_in[i];
  }

  float* d_in = nullptr;
  check(cudaMalloc(&d_in, static_cast<size_t>(n) * sizeof(float)), "cudaMalloc");
  check(cudaMemcpy(d_in, h_in, static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice), "H2D");

  const int threads = BLOCK;
  const int blocks = (n + threads - 1) / threads;
  float* d_partial = nullptr;
  check(cudaMalloc(&d_partial, static_cast<size_t>(blocks) * sizeof(float)), "partial");

  reduce_sum_kernel<<<blocks, threads>>>(d_in, d_partial, n);
  check(cudaGetLastError(), "kernel");
  check(cudaDeviceSynchronize(), "sync");

  float* h_partial = static_cast<float*>(std::malloc(static_cast<size_t>(blocks) * sizeof(float)));
  check(cudaMemcpy(h_partial, d_partial, static_cast<size_t>(blocks) * sizeof(float),
                   cudaMemcpyDeviceToHost),
        "D2H partial");

  float gpu_sum = 0.f;
  for (int b = 0; b < blocks; ++b) {
    gpu_sum += h_partial[b];
  }

  std::printf("CPU sum=%.1f GPU sum=%.1f (expect match)\n", cpu_sum, static_cast<double>(gpu_sum));

  cudaFree(d_in);
  cudaFree(d_partial);
  std::free(h_in);
  std::free(h_partial);
  return 0;
}
