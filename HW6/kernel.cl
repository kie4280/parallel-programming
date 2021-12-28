

__kernel void convolution(int filterWidth, int imageHeight, int imageWidth,
                          __global float *filter, __global float *inputImage,
                          __global float *outputImage) {
  int row = get_global_id(1);
  int col = get_global_id(0);

  int half_filter = filterWidth / 2;
  float sum = 0.0f;
    for (int a = -half_filter; a <= half_filter; ++a) {
      for (int b = -half_filter; b <= half_filter; ++b) {
        int x = col + b;
        int y = row + a;
        if (x < imageWidth && y < imageHeight && x >= 0 && y >= 0) {
          sum += filter[(a + half_filter) * filterWidth + (b + half_filter)] *
                 inputImage[y * imageWidth + x];
        }
      }
    }
  if (row < imageHeight && col < imageWidth) {
    outputImage[row * imageWidth + col] = sum;
  }
}
