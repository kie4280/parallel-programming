#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <random>

typedef long long int ll;

struct xorshift128p_state {
  uint64_t x[2];
};

/* The state must be seeded so that it is not all zero */
uint64_t xorshift128p(struct xorshift128p_state *state) {
  uint64_t t = state->x[0];
  uint64_t const s = state->x[1];
  state->x[0] = s;
  t ^= t << 23;       // a
  t ^= t >> 18;       // b -- Again, the shifts and the multipliers are tunable
  t ^= s ^ (s >> 5);  // c
  state->x[1] = t;
  return t + s;
}

int main(int argc, char **argv) {
  // --- DON'T TOUCH ---
  MPI_Init(&argc, &argv);
  double start_time = MPI_Wtime();
  double pi_result;
  ll tosses = atoi(argv[1]);
  int world_rank, world_size;
  // ---

  // TODO: MPI init
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  xorshift128p_state rs;
  std::random_device rd;
  rs.x[0] = world_rank;
  rs.x[1] = rd();

  // TODO: binary tree redunction

  ll toss = tosses / world_size + 1;
  ll hit = 0;
  for (long long int i = 0; i < toss / 2; ++i) {
    uint64_t u = xorshift128p(&rs);
    uint32_t x = (uint32_t)(u & 0x00000000ffffffff),
             y = (uint32_t)((u & 0xffffffff00000000) >> 32);
    float x1 = (float)((x & 0xffff0000) >> 16) / 0x0000ffff,
          x2 = (float)(x & 0x0000ffff) / 0x0000ffff;
    float y1 = (float)((y & 0xffff0000) >> 16) / 0x0000ffff,
          y2 = (float)(y & 0x0000ffff) / 0x0000ffff;
    hit += (x1 * x1 + y1 * y1 < 1 ? 1 : 0) + (x2 * x2 + y2 * y2 < 1 ? 1 : 0);
  }

  int rank = world_rank;
  int recv_id = 1;
  while (rank % 2 == 0 && recv_id < world_size) {
    MPI_Send(&hit, 1, MPI_LONG_LONG_INT, world_rank, 0, MPI_COMM_WORLD);
    hit = 0;
    ll other;
    MPI_Status status;
    MPI_Recv(&other, 1, MPI_LONG_LONG_INT, world_rank + recv_id, 0,
             MPI_COMM_WORLD, &status);
    MPI_Recv(&hit, 1, MPI_LONG_LONG_INT, world_rank, 0,
             MPI_COMM_WORLD, &status);

    hit += other;
    rank /= 2;
    recv_id *= 2;
  }
  if (world_rank > 0) {
    MPI_Send(&hit, 1, MPI_LONG_LONG_INT, world_rank - recv_id, 0,
             MPI_COMM_WORLD);
  }

  if (world_rank == 0) {
    // TODO: PI result

    pi_result = (double)hit / (toss * world_size) * 4;

    // --- DON'T TOUCH ---
    double end_time = MPI_Wtime();
    printf("%lf\n", pi_result);
    printf("MPI running time: %lf Seconds\n", end_time - start_time);
    // ---
  }

  MPI_Finalize();
  return 0;
}
