/****************************************************************************
 *
 * omp-traffic.c - Biham-Middleton-Levine traffic model
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
 * gcc -std=c99 -Wall -Wpedantic omp-traffic.c -o omp-traffic
 *
 * Run with:
 * ./omp-traffic [nsteps [rho [N]]]
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

/* Translates bidimensional indexes to a monodimensional one. 
 * |i| is the column index.
 * |j| is the row index.
 * |n| is the number of columns (length of the rows). */
#define IDX(i, j, n) ((i) * (n) + (j))

typedef unsigned char cell_t;

/* Possible values stored in a grid cell. */
enum {
    EMPTY = 0,  /* empty cell            */
    LR,         /* left-to-right vehicle */
    TB          /* top-to-bottom vehicle */
};

/* Move all left-to-right vehicles that are not blocked. */
void horizontal_step(cell_t *cur, cell_t *next, int n) {
    int i;
    int j;

    #pragma omp parallel for collapse(2)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (cur[IDX(i, (j - 1 + n) % n, n)] == LR && cur[IDX(i, j, n)] == EMPTY) {
                next[IDX(i, j, n)] = LR;
            } else if (cur[IDX(i, j, n)] == LR && cur[IDX(i, (j + 1) % n, n)] == EMPTY) {
                next[IDX(i, j, n)] = EMPTY;
            } else {
                next[IDX(i, j, n)] = cur[IDX(i, j, n)];
            }
        }
    }
}

/* Move all top-to-bottom vehicles that are not blocked. */
void vertical_step(cell_t *cur, cell_t *next, int n) {
    int i;
    int j;

    #pragma omp parallel for collapse(2)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (cur[IDX((i - 1 + n) % n, j, n)] == TB && cur[IDX(i, j, n)] == EMPTY) {
                next[IDX(i, j, n)] = TB;
            } else if (cur[IDX(i, j, n)] == TB && cur[IDX((i + 1) % n, j, n)] == EMPTY) {
                next[IDX(i, j, n)] = EMPTY;
            } else {
                next[IDX(i, j, n)] = cur[IDX(i, j, n)];
            }
        }
    }
}

/* Initialize |grid| with vehicles with density |rho|. |rho| must be
 * in the range [0, 1] (rho = 0 means no vehicle, rho = 1 means that
 * every cell is occupied by a vehicle). The direction is chosen with
 * equal probability. */
void setup(cell_t *grid, int n, float rho) {
    int i;
    int j;
    
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
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
 * |filename|. LR vehicles are shown as blue pixels, while TB vehicles
 * are shown in red. Empty cells are white. */
void dump(const cell_t *grid, int n, const char *filename) {
    int i;
    int j;
    FILE *out = fopen( filename, "w" );
    if (out == NULL) {
        printf("Cannot create \"%s\"\n", filename);
        abort();
    }
    fprintf(out, "P6\n");
    fprintf(out, "%d %d\n", n, n);
    fprintf(out, "255\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            switch( grid[IDX(i, j, n)] ) {
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
                printf("Error: unknown cell state %u\n", grid[IDX(i, j, n)]);
                abort();
            }
        }
    }
    fclose(out);
}

int main(int argc, char *argv[]) {
    cell_t *cur;
    cell_t *next;
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

    const size_t size = N * N * sizeof(cell_t);

    /* Allocate grids. */
    cur = (cell_t*) malloc(size);
    next = (cell_t*) malloc(size);

    setup(cur, N, rho);

    /* Dump the initialized grid. */
    snprintf(buf, BUFLEN, "omp-traffic-start.ppm");
    dump(cur, N, buf);

    tstart = hpc_gettime();
    for (s = 0; s < nsteps; s++) {
        horizontal_step(cur, next, N);
        vertical_step(next, cur, N);

	/* Dump each step. */
	/*snprintf(buf, BUFLEN, "omp-traffic-%05d.ppm", s);
	dump(cur, N, buf);*/
    }
    tend = hpc_gettime();

    fprintf(stderr, "Execution time (s): %f\n", tend - tstart);
    
    /* Dump the last state. */
    snprintf(buf, BUFLEN, "omp-traffic-%05d.ppm", s);
    dump(cur, N, buf);

    /* Free memory. */
    free(cur);
    free(next);
    
    return 0;
}
