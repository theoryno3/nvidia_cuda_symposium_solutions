
BIN=x.simple_add_blocks
NVCC=nvcc
NVOPTS=-arch sm_20

$(BIN): kernel.o
	$(NVCC) $(NVOPTS) -o $(BIN) kernel.o

kernel.o: kernel.cu
	$(NVCC) $(NVOPTS) -c kernel.cu 

clean:
	rm -rf kernel.o $(BIN)
