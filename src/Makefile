##############################################################################
##
## Autore: Moreno Marzolla
## Ultimo aggiornamento: 24 maggio 2017
##
## ---------------------------------------------------------------------------
##
## Questo Makefile e' predisposto per compilare le versioni OpenMP,
## MPI e CUDA del progetto. Ricordo che e' richiesto di sviluppare:
## (1) la versione OpenMP, (2) una A SCELTA tra MPI e CUDA. Questo
## Makefile si aspetta che i file abbiano nome:
##
## - omp-traffic.c per la versione OpenMP;
## - mpi-traffic.c per la versione MPI;
## - cuda-traffic.cu per la versione CUDA;
## - traffic.c per la versione seriale (schema fornito)
##
## Tutte le versioni, ad eccezione di quella CUDA, vengono compilate
## con i flag -std=c99 -Wall -Wpedantic.
##
## L'uso di un Makefile non e' obbligatorio.
##
##############################################################################
EXE_OMP:=$(basename $(wildcard omp-*.c))
EXE_MPI:=$(basename $(wildcard mpi-*.c))
EXE_CUDA:=$(basename $(wildcard cuda-*.cu))
EXE_SERIAL:=traffic
EXE:=$(EXE_OMP) $(EXE_MPI) $(EXE_SERIAL) $(EXE_CUDA)
CFLAGS+=-std=c99 -Wall -Wpedantic
LDLIBS+=-lrt
NVCC:=nvcc
NVCFLAGS:=-Wno-deprecated-gpu-targets

.PHONY: clean

ALL: $(EXE)

% : %.cu
	$(NVCC) $(NVCFLAGS) $< -o $@

$(EXE_OMP): cc=gcc
$(EXE_OMP): CFLAGS+=-fopenmp
$(EXE_OMP): LDLIBS+=-lgomp
openmp: $(EXE_OMP)

$(EXE_MPI): CC=mpicc
mpi: $(EXE_MPI)

cuda: $(EXE_CUDA)

clean:
	\rm -f *.ppm $(EXE) *.o *~
