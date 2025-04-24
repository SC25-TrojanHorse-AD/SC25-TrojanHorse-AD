include ../../../../../make.inc
pangulu_0201000.o:pangulu_cuda.cu
	$(NVCC) $(NVCCFLAGS) $(METIS_INC) -I/home/nyd/pkg_install/openmpi/include -Xcompiler -fPIC  -c $< -o $@
	mv $@ ../../../../../lib