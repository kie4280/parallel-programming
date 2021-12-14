#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#define blockWidth 8
#define blockHeight 8

template <typename T>
void print(T a) {
  std::cout << a << std::endl;
}

__device__ int mandel(float c_re, float c_im, int maxIteration) {
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < maxIteration; ++i) {
    if (z_re * z_re + z_im * z_im > 4.f) break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}

__global__ void mandelKernel(float upperX, float upperY, float lowerX,
                             float lowerY, int* buf, int resX, int resY,
                             float stepX, float stepY, int maxIterations) {
  // To avoid error caused by the floating number, use the following pseudo code
  //

  int rowX = blockIdx.x * blockDim.x + threadIdx.x;
  int rowY = blockIdx.y * blockDim.y + threadIdx.y;

  float x = lowerX + rowX * stepX;
  float y = lowerY + rowY * stepY;
  // __syncthreads();
  if (rowX >= resX || rowY >= resY) {
    return;
  }
  buf[rowY * resX + rowX] = mandel(x, y, maxIterations);
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int* img,
            int resX, int resY, int maxIterations) {
  float stepX = (upperX - lowerX) / resX;
  float stepY = (upperY - lowerY) / resY;

  int block_x = resX / blockWidth + 1;
  int block_y = resY / blockHeight + 1;

  int* buf = (int*)malloc(resX * resY * sizeof(int));
  int* cudaMem;
  dim3 TB(blockWidth, blockHeight);
  dim3 BG(block_x, block_y);
  cudaMalloc((void**)&cudaMem,
             block_x * block_y * blockWidth * blockHeight * sizeof(int));
  mandelKernel<<<BG, TB>>>(upperX, upperY, lowerX, lowerY, cudaMem, resX, resY,
                           stepX, stepY, maxIterations);
  cudaMemcpy(buf, cudaMem, resX * resY * sizeof(int), cudaMemcpyDeviceToHost);

  for (int a = 0; a < resX * resY; ++a) {
    img[a] = buf[a];
  }

  cudaFree(cudaMem);
  free(buf);
}
