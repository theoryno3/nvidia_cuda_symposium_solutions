
BIN=x.matmul_CUBLAS
NVCC=nvcc
NVOPTS=-O3 -arch sm_20

$(BIN): kernel.o
	$(NVCC) $(NVOPTS) -o $(BIN) kernel.o -lcublas

kernel.o: kernel.cu
	$(NVCC) $(NVOPTS) -c kernel.cu 

clean:
	rm -rf kernel.o $(BIN)
