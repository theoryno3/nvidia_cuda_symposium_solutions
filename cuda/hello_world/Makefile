

NVCC=nvcc
NVOPTS=-arch sm_20

hello_world: kernel.o
	$(NVCC) $(NVOPTS) -o x.hello_world kernel.o

kernel.o: kernel.cu
	$(NVCC) $(NVOPTS) -c kernel.cu 

clean:
	rm -rf kernel.o x.hello_world
