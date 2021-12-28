#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#define threadsPerBlock 32
#define pixelPerThread 4

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
                             int resY, float stepX, float stepY, size_t pitch,
                             int maxIterations) {
  // To avoid error caused by the floating number, use the following pseudo code
  //

  int rowX = blockIdx.x * blockDim.x + threadIdx.x;
  int rowY = blockIdx.y * blockDim.y + threadIdx.y;
  int res[pixelPerThread];
  int basey = rowY * pixelPerThread;

  float x = lowerX + rowX * stepX;
  for (int a = 0; a < pixelPerThread; ++a) {
    int inx = (basey + a);
    float y = lowerY + inx * stepY;

    res[a] = mandel(x, y, maxIterations);
  }
    __syncthreads();
  for (int a = 0; a < pixelPerThread; ++a) {
    *((int *)((char *)buf + (basey + a) * pitch) + rowX) = res[a];
  }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int *img,
            int resX, int resY, int maxIterations) {
  float stepX = (upperX - lowerX) / resX;
  float stepY = (upperY - lowerY) / resY;

  int *buf, *devMem;
  int block_x = resX / threadsPerBlock + 1;
  int block_y = resY / pixelPerThread + 1;

  size_t pitch;
  dim3 TB(threadsPerBlock, 1);
  dim3 GB(block_x, block_y);


  cudaMallocPitch((void **)&devMem, &pitch,
                  block_x * threadsPerBlock * sizeof(int),
                  block_y * pixelPerThread);
  mandelKernel<<<GB, TB>>>(lowerX, lowerY, devMem, resX, resY, stepX, stepY,
                           pitch, maxIterations);
  cudaHostAlloc(&buf, resX * resY * sizeof(int), cudaHostAllocDefault);
  cudaMemcpy2D(buf, sizeof(int) * resX, devMem, pitch, resX * sizeof(int), resY,
               cudaMemcpyDeviceToHost);
  // cudaDeviceSynchronize();

  for (int a = 0; a < resX * resY; ++a) {
    img[a] = buf[a];
  }

  cudaFree(devMem);
  cudaFree(buf);
  // free(buf);
}
