#include <mpi.h>

#include <iostream>
#include <string>

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

  if (world_rank == 0) {
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

    MPI_Bcast(*a_mat_ptr, n * m, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(*b_mat_ptr, m * l, MPI_INT, 0, MPI_COMM_WORLD);

  } else {
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&l, 1, MPI_INT, 0, MPI_COMM_WORLD);
    *n_ptr = n;
    *m_ptr = m;
    *l_ptr = l;
    *a_mat_ptr = new int[n * m];
    *b_mat_ptr = new int[m * l];
    MPI_Bcast(*a_mat_ptr, n * m, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(*b_mat_ptr, m * l, MPI_INT, 0, MPI_COMM_WORLD);
  }
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
  // std::cout << "greatings " << world_rank << " " << n << " " << m << " " << l
  //           << std::endl;
  int rows = n / world_size + 1;
  int row_start = world_rank * rows;
  int row_end = (world_rank + 1) * rows;
  row_end = row_end > n ? n : row_end;
  int elements = l * (row_end - row_start);
  int *rr = new int[elements]();

  for (int a = row_start; a < row_end; ++a) {
    for (int i = 0; i < l; ++i) {
      for (int k = 0; k < m; ++k) {
        rr[(a - row_start) * l + i] += a_mat[a * m + k] * b_mat[i + m * k];
      }
    }
  }
  // std::cout << "mult complete" << std::endl;

  int *result_buf = nullptr;
  MPI_Request *requests = nullptr;

  if (world_rank == 0) {
    result_buf = new int[n * l];
    int *handle = result_buf;
    requests = new MPI_Request[world_size];

    for (int a = 0; a < world_size; ++a) {
      int row_start = world_rank * rows;
      int row_end = (world_rank + 1) * rows;
      row_end = row_end > n ? n : row_end;
      int elements = l * (row_end - row_start);

      MPI_Irecv(handle, elements, MPI_INT, a, 0, MPI_COMM_WORLD, requests + a);
      handle += elements;
      // std::cout << "irecv" << std::endl;
    }
  }

  MPI_Send(rr, elements, MPI_INT, 0, 0, MPI_COMM_WORLD);
  // std::cout << "send complete" << std::endl;

  if (world_rank == 0) {
    MPI_Status *status = new MPI_Status[world_size];
    MPI_Waitall(world_size, requests, status);

    for (int a = 0; a < n; ++a) {
      // std::cout << "nonblocking recv" << std::endl;

      for (int b = 0; b < l; ++b) {
        std::cout << result_buf[a * l + b] << " ";
      }
      std::cout << std::endl;
    }

    delete[] result_buf;
    delete[] requests;
    delete[] status;
  }

  delete[] rr;
}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat) {
  delete[] a_mat;
  delete[] b_mat;
}
