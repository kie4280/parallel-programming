#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#define blockWidth 32
#define blockHeight 4

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

__global__ void mandelKernel(float lowerX, float lowerY, int *buf, int resX,
                             int resY, int offset_x, int offset_y, float stepX,
                             float stepY, size_t pitch, int maxIterations) {
  // To avoid error caused by the floating number, use the following pseudo code
  //

  int rowX = blockIdx.x * blockDim.x + threadIdx.x;
  int rowY = blockIdx.y * blockDim.y + threadIdx.y;
  // int res[4];



#pragma unroll
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      int col = rowX + offset_x * b;
      int row = rowY + offset_y * a;
      // if (col < resX && row < resY) {
      float x = lowerX + col * stepX;
      float y = lowerY + row * stepY;
      *((int *)((char *)buf + (row)*pitch) + (col)) = mandel(x, y, maxIterations);
      
      // }
    }
  }

  // __syncthreads();
  // *((int *)((char *)buf + (rowY)*pitch) + (rowX)) = res[0];

  // *((int *)((char *)buf + (rowY)*pitch) + (rowX + offset_x)) = res[1];
  // *((int *)((char *)buf + (rowY + offset_y) * pitch) + (rowX)) = res[2];
  // *((int *)((char *)buf + (rowY + offset_y) * pitch) + (rowX + offset_x)) =
  //     res[3];

  // for (int a = 0; a < 2; ++a) {
  //   int y = rowY + offset_y * a;
  //   int *row = ((int *)((char *)buf + y * pitch));
  //   for (int b = 0; b < 2; ++b) {
  //     int x = rowX + offset_x * b;
  //      row[x] = res[2 * a + b];
  //   }
  // }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int *img,
            int resX, int resY, int maxIterations) {
  float stepX = (upperX - lowerX) / resX;
  float stepY = (upperY - lowerY) / resY;

  int offset_x = resX / 2 + 1, offset_y = resY / 2 + 1;
  int block_x = offset_x / blockWidth + 1;
  int block_y = offset_y / blockHeight + 1;

  int *buf, *devMem;

  dim3 TB(blockWidth, blockHeight);
  dim3 GB(block_x, block_y);
  size_t pitch;
  cudaMallocPitch((void **)&devMem, &pitch,
                  block_x * blockWidth * 2 * sizeof(int),
                  block_y * 2 * blockHeight);

  mandelKernel<<<GB, TB>>>(lowerX, lowerY, devMem, resX, resY, offset_x,
                           offset_y, stepX, stepY, pitch, maxIterations);
  cudaHostAlloc(&buf, resX * resY * sizeof(int), cudaHostAllocDefault);
  cudaMemcpy2D(buf, sizeof(int) * resX, devMem, pitch, resX * sizeof(int), resY,
               cudaMemcpyDeviceToHost);

  for (int a = 0; a < resX * resY; ++a) {
    img[a] = buf[a];
  }

  cudaFree(devMem);
  cudaFreeHost(buf);
}
