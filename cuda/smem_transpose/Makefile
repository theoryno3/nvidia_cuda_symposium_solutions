
BIN=x.smem_transpose
NVCC=nvcc
NVOPTS=-O3 -arch sm_20 -lineinfo

$(BIN): kernel.o
	$(NVCC) $(NVOPTS) -o $(BIN) kernel.o

kernel.o: kernel.cu
	$(NVCC) $(NVOPTS) -c kernel.cu 

clean:
	rm -rf kernel.o $(BIN)
