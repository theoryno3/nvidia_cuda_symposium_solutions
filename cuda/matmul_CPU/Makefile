
BIN=x.matmul_CPU
NVCC=nvcc
NVOPTS=-O3 -arch sm_20

$(BIN): kernel.o
	$(NVCC) $(NVOPTS) -o $(BIN) kernel.o

kernel.o: kernel.cu
	$(NVCC) $(NVOPTS) -c kernel.cu 

clean:
	rm -rf kernel.o $(BIN)
