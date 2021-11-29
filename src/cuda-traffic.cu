/****************************************************************************
 *
 * cuda-traffic.c - Biham-Middleton-Levine traffic model
 *
 * Luka Micheletti, Matricola 723450
 * Progetto HPC 2016/2017
 *
 * ---------------------------------------------------------------------------
 *
 * This program implements the Biham-Middleton-Levine traffic model
 * The BML traffic model is a simple three-state 2D cellular automaton
 * over a toroidal square lattice space. Initially, each cell is
 * either empty, or contains a left-to-right (LR) or top-to-bottom
 * (TB) moving vehicle. The model evolves at discrete time steps. Each
 * step is logically divided into two phases: in the first phase only
 * LR vehicles move, provided that the destination cell is empty; in
 * the second phase, only TB vehicles move, again provided that the
 * destination cell is empty.
 *
 * Compile with:
 * nvcc -Wno-deprecated-gpu-targets cuda-traffic.cu -o cuda-traffic
 *
 * Run with:
 * ./cuda-traffic [nsteps [rho [N]]]
 * 
 * where nsteps is the number of simulation steps to execute, rho is
 * the density of vehicles (probability that a cell is occupied by a
 * vehicle), and N is the grid size.
 *
 ****************************************************************************/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>

#define BUFLEN 256
#define BLKSIZE 16

/* Translates bidimensional indexes to a monodimensional one. 
 * |i| is the column index.
 * |j| is the row index.
 * |n| is the number of columns (length of the rows). */
#define IDX(i, j, n) ((i) * (n) + (j))

typedef unsigned char cell_t;

/* Possible values stored in a grid cell */
enum {
    EMPTY = 0,  /* empty cell            */
    LR,         /* left-to-right vehicle */
    TB          /* top-to-bottom vehicle */
};

/*|grid| points to a (n + 2) * (n + 2) block of bytes; this function copies
  the bottom and top n elements to the opposite ghost cell layer (see figure
  below). 
 
   0 1              n n+1
   | |              | |
   v v              v v
  +-+----------------+-+
  |Y|YYYYYYYYYYYYYYYY|Y| <- 0
  +-+----------------+-+
  |X|XXXXXXXXXXXXXXXX|X| <- 1
  |\|                |\|
  |\|                |\|
  |\|                |\|
  |\|                |\|
  |Y|YYYYYYYYYYYYYYYY|Y| <- n
  +-+----------------+-+
  |X|XXXXXXXXXXXXXXXX|X| <- n+1
  +-+----------------+-+
 */
__global__ void copy_top_bottom(cell_t *grid, int n) {
    const int j = 1 + threadIdx.x + blockIdx.x * blockDim.x;
    if (j < n + 1) {
        grid[IDX(n + 1, j, n + 2)] = grid[IDX(1, j, n + 2)];
        grid[IDX(0, j, n + 2)] = grid[IDX(n, j, n + 2)];
    }    
}

/*|grid| points to a (n + 2) * (n + 2) block of bytes; this function copies
  the left and right (n + 2) elements to the opposite ghost cell layer (see figure
  below).
 
   0 1              n n+1
   | |              | |
   v v              v v
  +-+----------------+-+
  |Y|X\\\\\\\\\\\\\\Y|X| <- 0
  +-+----------------+-+
  |Y|X              Y|X| <- 1
  |Y|X              Y|X|
  |Y|X              Y|X|
  |Y|X              Y|X|
  |Y|X              Y|X|
  |Y|X              Y|X| <- n
  +-+----------------+-+
  |Y|X\\\\\\\\\\\\\\Y|X| <- n+1
  +-+----------------+-+
 */
__global__ void copy_left_right(cell_t *grid, int n) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n + 2) {
        grid[IDX(i, n + 1, n + 2)] = grid[IDX(i, 1, n + 2)];
        grid[IDX(i, 0, n + 2)] = grid[IDX(i, n, n + 2)];
    }
}

/* Move all left-to-right vehicles that are not blocked. */
__global__ void horizontal_step(cell_t *cur, cell_t *next, int n) {
    const int x = 1 + threadIdx.y + blockIdx.y * blockDim.y;
    const int y = 1 + threadIdx.x + blockIdx.x * blockDim.x;

    if (x < n + 1 && y < n + 1) {
        if (cur[IDX(x, y - 1, n + 2)] == LR && cur[IDX(x, y, n + 2)] == EMPTY) {
            next[IDX(x, y, n + 2)] = LR;
        } else if (cur[IDX(x, y, n + 2)] == LR && cur[IDX(x, y + 1, n + 2)] == EMPTY) {
            next[IDX(x, y, n + 2)] = EMPTY;
        } else {
            next[IDX(x, y, n + 2)] = cur[IDX(x, y, n + 2)];
        }
	}
}

/* Move all top-to-bottom vehicles that are not blocked. */
__global__ void vertical_step(cell_t *cur, cell_t *next, int n) {
    const int x = 1 + threadIdx.y + blockIdx.y * blockDim.y;
    const int y = 1 + threadIdx.x + blockIdx.x * blockDim.x;

    if (x < n + 1 && y < n + 1) {
        if (cur[IDX(x - 1, y, n + 2)] == TB && cur[IDX(x, y, n + 2)] == EMPTY) {
            next[IDX(x, y, n + 2)] = TB;
        } else if (cur[IDX(x, y, n + 2)] == TB && cur[IDX(x + 1, y, n + 2)] == EMPTY) {
            next[IDX(x, y, n + 2)] = EMPTY;
        } else {
            next[IDX(x, y, n + 2)] = cur[IDX(x, y, n + 2)];
        }
    }
}

/* Initialize |grid| with vehicles with density |rho|. |rho| must be
   in the range [0, 1] (rho = 0 means no vehicle, rho = 1 means that
   every cell is occupied by a vehicle). The direction is chosen with
   equal probability. */
void setup(cell_t *grid, int n, float rho) {
    int i;
    int j;

    for (i = 1; i < n - 1; i++) {
        for (j = 1; j < n - 1; j++) {
            if (((float) rand() / (float) (RAND_MAX)) < rho) {
                if (rand() % 100 < 50) {
                    grid[IDX(i, j, n)] = LR;
                } else {
                    grid[IDX(i, j, n)] = TB;
                }
            } else {
                grid[IDX(i, j, n)] = EMPTY;
            }
        }
    }
}

/* Dump |grid| as a PPM (Portable PixMap) image written to file
   |filename|. LR vehicles are shown as blue pixels, while TB vehicles
   are shown in red. Empty cells are white. */
void dump(const cell_t *grid, int n, const char *filename) {
    int i;
    int j;
    FILE *out = fopen( filename, "w" );
    if (out == NULL) {
        printf("Cannot create \"%s\"\n", filename);
        abort();
    }
    fprintf(out, "P6\n");
    fprintf(out, "%d %d\n", n - 2, n - 2);
    fprintf(out, "255\n");
    for (i = 1; i < n - 1; i++) {
        for (j = 1; j < n - 1; j++) {
            switch(grid[IDX(i, j, n)]) {
            case EMPTY:
                fprintf(out, "%c%c%c", 255, 255, 255);
                break;
            case TB:
                fprintf(out, "%c%c%c", 0, 0, 255);
                break;
            case LR:
                fprintf(out, "%c%c%c", 255, 0, 0);
                break;
            default:
                /*printf("Error: unknown cell state %u\nIndex: %d-%d\n", grid[IDX(i, j, n)], i - 1, j - 1);
                abort();*/
                fprintf(out, "%c%c%c", grid[IDX(i, j, n)], 0, 255 - grid[IDX(i, j, n)]);
            }
        }
    }
    fclose(out);
}

int main(int argc, char *argv[]) {
    cell_t *grid;

    cell_t *d_cur;
    cell_t *d_next;

    char buf[BUFLEN];
    int s;
    int N = 256;
    int nsteps = 512;
    float rho = 0.2;
    double tstart;
    double tend;

    if (argc > 4) {
        printf("Usage: %s [nsteps [rho [N]]]\n", argv[0]);
        return -1;
    }

    if (argc > 1) {
        nsteps = atoi(argv[1]);
    }

    if (argc > 2) {
        rho = atof(argv[2]);
    }

    if (argc > 3) {
        N = atoi(argv[3]);
    }

    /* |size| comprehends a layer of ghost cells. */
    const size_t size = (N + 2) * (N + 2) * sizeof(cell_t);

    /* Define block size and grid size for copying ghost cells on the sides of the grid. */
    dim3 cpy_block(BLKSIZE);
    dim3 cpy_grid((N + 2 + BLKSIZE - 1) / BLKSIZE);

    /* Define block size and grid size for calculating the steps. */
    dim3 step_block(BLKSIZE, BLKSIZE);
    dim3 step_grid((N + BLKSIZE - 1) / BLKSIZE, (N + BLKSIZE - 1) / BLKSIZE);

    /* Allocate grids. */
    grid = (cell_t*) malloc(size);

    cudaMalloc((void **) &d_cur, size);
    cudaMalloc((void **) &d_next, size);


    setup(grid, N + 2, rho);

    /* Dump the initialized grid. */
    snprintf(buf, BUFLEN, "cuda-traffic-start.ppm");
    dump(grid, N + 2, buf);

    /* Copy the initialized grid to the device. */
    cudaMemcpy(d_cur, grid, size, cudaMemcpyHostToDevice);

    tstart = hpc_gettime();
    for (s = 0; s < nsteps; s++) {
        /* Initialize the ghost cells in the first grid. */
        copy_top_bottom<<<cpy_grid, cpy_block>>>(d_cur, N);
        copy_left_right<<<cpy_grid, cpy_block>>>(d_cur, N);

        horizontal_step<<<step_grid, step_block>>>(d_cur, d_next, N);

    	  /* Initialize the ghost cells in the second grid. */
    	  copy_top_bottom<<<cpy_grid, cpy_block>>>(d_next, N);
    	  copy_left_right<<<cpy_grid, cpy_block>>>(d_next, N);

        vertical_step<<<step_grid, step_block>>>(d_next, d_cur, N);

        /* Dump each step. */
        /*cudaMemcpy(grid, d_cur, size, cudaMemcpyDeviceToHost);
        snprintf(buf, BUFLEN, "cuda-traffic-%05d.ppm", s);
        dump(grid, N + 2, buf);*/
    }
    cudaDeviceSynchronize();
    tend = hpc_gettime();
    fprintf(stderr, "Execution time (s): %f\n", tend - tstart);

    /* Copy the result grid back to the host. */
    cudaMemcpy(grid, d_cur, size, cudaMemcpyDeviceToHost);

    /* Dump the last state. */
    snprintf(buf, BUFLEN, "cuda-traffic-%05d.ppm", s);
    dump(grid, N + 2, buf);

    /* Free memory. */
    free(grid);

    cudaFree(d_cur);
    cudaFree(d_next);

    return 0;
}
