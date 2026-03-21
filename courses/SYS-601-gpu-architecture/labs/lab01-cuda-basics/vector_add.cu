// Minimal CUDA vector add — Lab 01 starter.
// Build: nvcc -std=c++17 -O2 vector_add.cu -o vector_add

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
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
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);

  float *h_a = static_cast<float*>(std::malloc(bytes));
  float *h_b = static_cast<float*>(std::malloc(bytes));
  float *h_c = static_cast<float*>(std::malloc(bytes));
  for (int i = 0; i < n; ++i) {
    h_a[i] = 1.0f;
    h_b[i] = 2.0f;
  }

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  check(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
  check(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
  check(cudaMalloc(&d_c, bytes), "cudaMalloc d_c");

  check(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice), "H2D a");
  check(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice), "H2D b");

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  vector_add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
  check(cudaGetLastError(), "kernel");
  check(cudaDeviceSynchronize(), "sync");

  check(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost), "D2H c");

  float err = 0.f;
  for (int i = 0; i < n; ++i) {
    err += std::abs(h_c[i] - 3.0f);
  }
  std::printf("sum abs error: %f (expect ~0)\n", err);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  std::free(h_a);
  std::free(h_b);
  std::free(h_c);
  return 0;
}
