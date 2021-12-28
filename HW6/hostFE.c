#include "hostFE.h"

#include <stdio.h>
#include <stdlib.h>

#include "helper.h"

#define BLOCK_WIDTH 8
#define BLOCK_HEIGHT 8

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program) {
  cl_int status;
  int filterSize = filterWidth * filterWidth;
  size_t imageBytes = sizeof(float) * imageHeight * imageWidth;
  cl_kernel kernel = clCreateKernel(*program, "convolution", &status);

  cl_command_queue sync_queue =
      clCreateCommandQueue(*context, *device, 0, &status);
  cl_mem input_img =
      clCreateBuffer(*context, CL_MEM_READ_WRITE, imageBytes, NULL, &status);
  cl_mem output_img =
      clCreateBuffer(*context, CL_MEM_READ_WRITE, imageBytes, NULL, &status);
  cl_mem filterBuf = clCreateBuffer(
      *context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(float) * filterWidth * filterWidth, filter, &status);

  clSetKernelArg(kernel, 0, sizeof(int), &filterWidth);
  clSetKernelArg(kernel, 1, sizeof(int), &imageHeight);
  clSetKernelArg(kernel, 2, sizeof(int), &imageWidth);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), &filterBuf);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), &input_img);
  clSetKernelArg(kernel, 5, sizeof(cl_mem), &output_img);

  status = clEnqueueWriteBuffer(sync_queue, input_img, CL_TRUE, 0, imageBytes,
                                (void *)inputImage, 0, NULL, NULL);
  size_t group_size[2] = {BLOCK_WIDTH * (imageWidth / BLOCK_WIDTH + 1),
                          BLOCK_HEIGHT * (imageHeight / BLOCK_HEIGHT + 1)};
  size_t work_size[2] = {BLOCK_WIDTH, BLOCK_HEIGHT};
  status = clEnqueueNDRangeKernel(sync_queue, kernel, 2, 0, &group_size,
                                  &work_size, 0, NULL, NULL);

  status = clEnqueueReadBuffer(sync_queue, output_img, CL_TRUE, 0, imageBytes,
                               (void *)outputImage, 0, NULL, NULL);
//   clReleaseKernel(kernel);
//   clReleaseMemObject(input_img);
//   clReleaseMemObject(output_img);
//   clReleaseMemObject(filterBuf);
//   clReleaseCommandQueue(sync_queue);
}