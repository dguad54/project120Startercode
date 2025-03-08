#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <mma.h>
#include <sstream>
#include <stdio.h>
#include <torch/extension.h>
#include <vector>

__global__ void matTransKernel(float* AT, float* A, int N);

void matTrans(torch::Tensor AT, torch::Tensor A)  {
  assert(AT.size(0) == AT.size(1));
  assert(AT.size(0) == A.size(0));
  assert(AT.size(1) == A.size(1));
  matTransKernel<<<1, 512>>>(AT.data_ptr<float>(), A.data_ptr<float>(), A.size(0));
}

__global__ void matTransKernel(float* AT, float* A, int N)  {
  int tid = blockIdx.x*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x;
  for(int i = tid; i < N*N; i += blockDim.x*gridDim.x*blockDim.y) {
        int row = i / N;
        int col = i % N;
        AT[col*N+row] = A[i];
  }
}
