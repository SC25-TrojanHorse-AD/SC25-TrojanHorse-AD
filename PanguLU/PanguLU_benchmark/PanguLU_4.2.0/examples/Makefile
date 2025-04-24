# PATH_TO_METIS_I64_LIB = /home/nyd/pkg_install/metis/i64/lib
# PATH_TO_GKLIB_LIB = /home/nyd/pkg_install/gklib/lib
# PATH_TO_OPENBLAS_LIB = /home/nyd/pkg_install/openblas/lib
LINK_METIS = $(PATH_TO_METIS_I64_LIB)/libmetis.a $(PATH_TO_GKLIB_LIB)/libGKlib.a
OPENBLAS_LIB = $(PATH_TO_OPENBLAS_LIB)/libopenblas.a
LINK_CUDA = -L/usr/local/cuda/lib64 -lcudart -lcusparse -lstdc++
LINK_PANGULU = ../lib/libpangulu.a # Derictly importing static library as compiler input makes dynamic library loader searching the directory of static library.

all: pangulu_example.elf

pangulu_example.elf:example.c
	#mpicc -g -fsanitize=address $< -DCALCULATE_TYPE_R64 -I../include $(LINK_PANGULU) $(LINK_CUDA) $(LINK_METIS) $(OPENBLAS_LIB) -fopenmp -lpthread -lm -o $@
	#mpicc -g $< -DCALCULATE_TYPE_R64 -I../include $(LINK_PANGULU) $(LINK_CUDA) $(LINK_METIS) $(OPENBLAS_LIB) -fopenmp -lpthread -lm -o $@
	mpicc -O3 $< -DCALCULATE_TYPE_R64 -I../include $(LINK_PANGULU) $(LINK_CUDA) $(LINK_METIS) $(OPENBLAS_LIB) -fopenmp -lpthread -lm -o $@

clean:
	rm -f *.elf
