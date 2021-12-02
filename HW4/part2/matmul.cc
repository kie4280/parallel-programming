#include <mpi.h>

#include <iostream>
#include <string>

#define DEBUG

template <typename T>
inline void print(T a) {
#ifdef DEBUG
  std::cout << a << std::endl;
#endif
}

// Read size of matrix_a and matrix_b (n, m, l) and whole data of matrixes from
// stdin
//
// n_ptr:     pointer to n
// m_ptr:     pointer to m
// l_ptr:     pointer to l
// a_mat_ptr: pointer to matrix a (a should be a continuous memory space for
// placing n * m elements of int) b_mat_ptr: pointer to matrix b (b should be a
// continuous memory space for placing m * l elements of int)
void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr,
                        int **b_mat_ptr) {
  int n, m, l;
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  // std::cout <<"get data" << std::endl;

  MPI_Request req;
  double start = MPI_Wtime();

  if (world_rank == 0) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    if (std::cin.good()) {
      std::cin >> n >> m >> l;
      *n_ptr = n;
      *m_ptr = m;
      *l_ptr = l;
      *a_mat_ptr = new int[n * m];
      *b_mat_ptr = new int[m * l];
      for (int a = 0; a < n; ++a) {
        for (int b = 0; b < m; ++b) {
          int i;
          std::cin >> i;
          (*a_mat_ptr)[a * m + b] = i;
        }
      }
      for (int a = 0; a < m; ++a) {
        for (int b = 0; b < l; ++b) {
          int i;
          std::cin >> i;
          (*b_mat_ptr)[a * l + b] = i;
        }
      }
    }

    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&l, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int rows = n / world_size;
    int scatter_size = rows * m;
    int *tmp = new int[scatter_size];
    MPI_Iscatter(*a_mat_ptr, scatter_size, MPI_INT, tmp, scatter_size, MPI_INT,
                 0, MPI_COMM_WORLD, &req);
    MPI_Bcast(*b_mat_ptr, m * l, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Status stat;
    MPI_Wait(&req, &stat);
    delete[] tmp;

  } else {
    // MPI_Status stat;
    // MPI_Recv(&m, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &stat);
    // MPI_Recv(&n, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &stat);
    // MPI_Recv(&l, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &stat);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&l, 1, MPI_INT, 0, MPI_COMM_WORLD);
    *n_ptr = n;
    *m_ptr = m;
    *l_ptr = l;
    int rows = n / world_size;
    int scatter_size = rows * m;

    *a_mat_ptr = new int[scatter_size];
    *b_mat_ptr = new int[m * l];
    // MPI_Recv(*a_mat_ptr, n * m, MPI_INT, 0, 3, MPI_COMM_WORLD, &stat);
    // MPI_Recv(*b_mat_ptr, m * l, MPI_INT, 0, 4, MPI_COMM_WORLD, &stat);
    MPI_Iscatter(0, 0, MPI_INT, *a_mat_ptr, scatter_size, MPI_INT, 0,
                 MPI_COMM_WORLD, &req);
    MPI_Bcast(*b_mat_ptr, m * l, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Status stat;
    MPI_Wait(&req, &stat);
  }
  start = MPI_Wtime() - start;

  print("get complete" + std::to_string(start));
}

// Just matrix multiplication (your should output the result in this function)
//
// n:     row number of matrix a
// m:     col number of matrix a / row number of matrix b
// l:     col number of matrix b
// a_mat: a continuous memory placing n * m elements of int
// b_mat: a continuous memory placing m * l elements of int
void matrix_multiply(const int n, const int m, const int l, const int *a_mat,
                     const int *b_mat) {
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  double start = MPI_Wtime();

  int rows = n / world_size;
  int elements = rows * l;

  int *rr = new int[elements]();
  // int *B_trans = new int[l*m];
  // for (int a=0; a<m; ++a) {
  //   for (int b=0; b<l; ++b) {
  //     B_trans[a*l+b] = b_mat[]
  //   }
  // }

  for (int a = 0; a < rows; ++a) {
    int a2 = a * m;
    for (int i = 0; i < l; ++i) {
      int a1 = a * l + i;
      for (int k = 0; k < m; ++k) {
        rr[a1] += a_mat[a2 + k] * b_mat[i + l * k];
      }
    }
  }
  start = MPI_Wtime() - start;

  print("mult complete " + std::to_string(start));

  int *result_buf = nullptr;
  MPI_Request req;

  if (world_rank == 0) {
    result_buf = new int[n * l]();
  }
  // std::cout << "send" << std::endl;
  MPI_Igather(rr, elements, MPI_INT, result_buf, elements, MPI_INT, 0,
              MPI_COMM_WORLD, &req);
  print("gather");
  if (world_rank == 0) {
    for (int a = rows * world_size; a < n; ++a) {
      int a2 = a * m;
      for (int i = 0; i < l; ++i) {
        int a1 = a * l + i;
        for (int k = 0; k < m; ++k) {
          result_buf[a1] += a_mat[a2 + k] * b_mat[i + l * k];
        }
      }
    }
    MPI_Status stat;

    MPI_Wait(&req, &stat);

    for (int a = 0; a < n; ++a) {
      for (int b = 0; b < l; ++b) {
#ifndef DEBUG
        std::cout << result_buf[a * l + b] << " ";
#endif
      }
#ifndef DEBUG
      std::cout << std::endl;
#endif
    }
  } else {
    MPI_Status stat;

    MPI_Wait(&req, &stat);
  }

  if (world_rank == 0) {
    delete[] result_buf;
  }
  delete[] rr;
}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat) {
  delete[] a_mat;
  delete[] b_mat;
}
