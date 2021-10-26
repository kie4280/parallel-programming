#include <pthread.h>

#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

#define DEBUGGING
#define MAX_THREADS 256

typedef long long int LL;

template <typename T>
void print(T &input) {
#ifdef DEBUGGING
  std::cout << input << std::endl;
#endif
}

// ----------------definitions end--------------------
LL num_sim = 0;

std::random_device rd;

void *monte_carlo(void *args) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
  int index = *static_cast<int *>(args);
  LL inside_sum = 0;
  for (LL a = 0; a < num_sim; ++a) {
    float x = dis(gen), y = dis(gen);
    if (x * x + y * y <= 0.25f) {
      ++inside_sum;
    }
  }
  return (void *)inside_sum;
}

int main(int argc, char **argv) {
  int thread_num;
  LL tosses;
  if (argc != 3) return EXIT_FAILURE;
  thread_num = std::atoi(argv[1]);
  if (thread_num >= MAX_THREADS) {
    print("Too many threads");
    return EXIT_FAILURE;
  }
  tosses = std::atoll(argv[2]);
  num_sim = tosses / (thread_num + 1) + 1;
  pthread_t threads[MAX_THREADS];
  int thread_index[MAX_THREADS];
  thread_index[0] = 0;

  for (int a = 1; a <= thread_num; ++a) {
    thread_index[a] = a;
    pthread_create(&threads[a], nullptr, monte_carlo, &thread_index[a]);
  }

  LL sum = (LL)monte_carlo(&thread_index[0]);

  for (int a = 1; a <= thread_num; ++a) {
    void *result;
    pthread_join(threads[a], &result);
    LL n = (LL)result;
    sum += n;
    // print(n);
  }
  double out = (double)sum / (num_sim * (thread_num + 1)) * 4;
  print(out);

  return EXIT_SUCCESS;
}