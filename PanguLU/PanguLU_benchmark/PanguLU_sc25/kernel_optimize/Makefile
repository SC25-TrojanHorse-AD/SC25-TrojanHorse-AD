MAIN_CU := main.cu
EXEC := main
CFLAGS += -DPANGULU_PLATFORM_ENV

all: $(EXEC)

$(EXEC): $(MAIN_CU)
	nvcc -O0 -g -G $< -o $@.elf -I../include -I/usr/local/cuda/include -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61

clean:
	rm -f $(EXEC).elf
