include ../make.inc
all:pangulu_host

src:=$(wildcard *.c)
pangulu_host : pangulu_platforms $(src:.c=.o) 

%.o:%.c
	$(MPICC) $(MPICCFLAGS) $(METISFLAGS) $(PANGULU_FLAGS) $(SC25_INC) -c $< -o $@ -fPIC
	mv $@ ../lib

pangulu_platforms:
	cd .. && python3 build_helper.py generate_platform_helper
	cd .. && python3 build_helper.py compile_platform_code

clean:
	-(rm -f ../lib/pangulu*.o)
	-(rm -f ./pangulu*.o)
