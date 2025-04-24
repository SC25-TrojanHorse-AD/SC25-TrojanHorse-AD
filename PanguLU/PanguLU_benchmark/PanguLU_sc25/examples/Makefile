LINK_METIS = $(PATH_TO_METIS_I64_LIB)/libmetis.a $(PATH_TO_GKLIB_LIB)/libGKlib.a
OPENBLAS_LIB = $(PATH_TO_OPENBLAS_LIB)/libopenblas.a
LINK_CUDA = -L/usr/local/cuda/lib64 -lcudart -lcusparse -lstdc++
LINK_PANGULU = ../lib/libpangulu.a # Derictly importing static library as compiler input makes dynamic library loader searching the directory of static library.
LINK_SC25 = $(PATH_TO_SC25_LIB)/libsc25.a

all: pangulu_example.elf

pangulu_example.elf:example.c
	mpicc -O3 $< -DCALCULATE_TYPE_R64 -I../include $(LINK_PANGULU) $(LINK_SC25) $(LINK_CUDA) $(LINK_METIS) $(OPENBLAS_LIB) -fopenmp -lpthread -lm -o $@

clean:
	rm -f *.elf
