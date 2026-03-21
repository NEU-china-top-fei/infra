// Tiled matrix multiply (float) using __shared__ tiles.
// Build: nvcc -std=c++17 -O2 matrix_mul.cu -o matrix_mul

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

constexpr int TILE = 16;

__global__ void matmul_tiled(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K) {
  __shared__ float tile_a[TILE][TILE];
  __shared__ float tile_b[TILE][TILE];

  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;

  float acc = 0.f;
  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
    const int a_col = t * TILE + threadIdx.x;
    const int a_row = row;
    tile_a[threadIdx.y][threadIdx.x] =
        (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.f;

    const int b_row = t * TILE + threadIdx.y;
    const int b_col = col;
    tile_b[threadIdx.y][threadIdx.x] =
        (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.f;

    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
      acc += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = acc;
  }
}

static void check(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
    std::exit(1);
  }
}

static void cpu_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float s = 0.f;
      for (int k = 0; k < K; ++k) {
        s += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = s;
    }
  }
}

int main() {
  const int M = 128, N = 96, K = 64;
  const size_t sa = static_cast<size_t>(M) * K * sizeof(float);
  const size_t sb = static_cast<size_t>(K) * N * sizeof(float);
  const size_t sc = static_cast<size_t>(M) * N * sizeof(float);

  float *h_a = static_cast<float*>(std::malloc(sa));
  float *h_b = static_cast<float*>(std::malloc(sb));
  float *h_c = static_cast<float*>(std::malloc(sc));
  float *h_ref = static_cast<float*>(std::malloc(sc));
  for (int i = 0; i < M * K; ++i) h_a[i] = static_cast<float>(rand() % 5) * 0.01f;
  for (int i = 0; i < K * N; ++i) h_b[i] = static_cast<float>(rand() % 5) * 0.01f;

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  check(cudaMalloc(&d_a, sa), "cudaMalloc A");
  check(cudaMalloc(&d_b, sb), "cudaMalloc B");
  check(cudaMalloc(&d_c, sc), "cudaMalloc C");
  check(cudaMemcpy(d_a, h_a, sa, cudaMemcpyHostToDevice), "H2D A");
  check(cudaMemcpy(d_b, h_b, sb, cudaMemcpyHostToDevice), "H2D B");

  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  matmul_tiled<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
  check(cudaGetLastError(), "kernel");
  check(cudaDeviceSynchronize(), "sync");
  check(cudaMemcpy(h_c, d_c, sc, cudaMemcpyDeviceToHost), "D2H C");

  cpu_matmul(h_a, h_b, h_ref, M, N, K);
  float err = 0.f;
  for (int i = 0; i < M * N; ++i) {
    err += std::abs(h_c[i] - h_ref[i]);
  }
  std::printf("matmul_tiled max-like err sum: %f\n", err);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  std::free(h_a);
  std::free(h_b);
  std::free(h_c);
  std::free(h_ref);
  return 0;
}
