############################################################################
#
#  Program:         SuperLU_DIST
#
#  Module:          make.inc
#
#  Purpose:         Top-level Definitions
#
#  Creation date:   March 1, 2016	version 5.0.0
#
#  Modified:	    October 13, 2017    version 5.2.1
#		    February 20, 2021   version 7.0.0
#		    October 5, 2021     version 7.1.0
#
############################################################################
#
#  The name of the libraries to be created/linked to
#
SuperLUroot = /usr/local
#DSUPERLULIB = $(SuperLUroot)/SRC/libsuperlu_dist.a
#DSUPERLULIB = $(SuperLUroot)/lib/libsuperlu_dist.a
DSUPERLULIB = /usr/local/lib/libsuperlu_dist.a
INCLUDEDIR  = $(SuperLUroot)/include

XSDK_INDEX_SIZE = 64
SLU_HAVE_LAPACK = FALSE
HAVE_PARMETIS   = TRUE
HAVE_COLAMD     = 
HAVE_COMBBLAS   = 
HAVE_CUDA       = TRUE
HAVE_HIP        = 

XSDK_ENABLE_Fortran = ON
ifeq ($(XSDK_ENABLE_Fortran),ON)
#  DFORTRANLIB = $(SuperLUroot)/lib/libsuperlu_dist_fortran.a
  DFORTRANLIB = /usr/local/lib/libsuperlu_dist_fortran.a
  LIBS = $(DFORTRANLIB) $(DSUPERLULIB) /home/yida/pkg_install/openblas_0.3.28/lib/libopenblas.a -lm
  LIBS +=  -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi -lgfortran -lm -lgcc_s -lgcc -lquadmath -lm -lc -lgcc_s -lgcc
else
  LIBS = $(DSUPERLULIB) /home/yida/pkg_install/openblas_0.3.28/lib/libopenblas.a -lm
endif

LIBS	 += 
LIBS	 += /home/yida/pkg_install/parmetis/i64/lib/libparmetis.a /home/yida/pkg_install/metis/i64/lib/libmetis.a /home/yida/pkg_install/gklib/lib/libGKlib.a
LIBS	 += 
LIBS 	 += 
LIBS 	 += -lgcc_s -lgcc -lc -lgcc_s -lgcc -lmpi 
#LIBS     += CUDA::cudart CUDA::cublas CUDA::cusolver

CUDALIBS = /usr/local/cuda/lib64/libcudart.so /usr/local/cuda/lib64/libcublas.so /usr/local/cuda/lib64/libcusolver.so /usr/local/cuda/lib64/libcusparse.so
LIBS     += $(CUDALIBS)

#
#  The archiver and the flag(s) to use when building archive (library)

#  If your system has no ranlib, set RANLIB = echo.
#
ARCH         = /usr/bin/ar
ARCHFLAGS    = cr
RANLIB       = /usr/bin/ranlib

CC           = /usr/local/bin/mpicc
CFLAGS 	     = -O3 -DNDEBUG -I/home/yida/pkg_install/parmetis/i64/include -I/home/yida/pkg_install/metis/i64/include -DUSE_VENDOR_BLAS -fopenmp   
##-fPIC
#CFLAGS      += -D
# CFLAGS     += 
CXX          = /usr/local/bin/mpicxx
CXXFLAGS     = -O3 -DNDEBUG -fopenmp  
NVCC	     = /usr/local/cuda/bin/nvcc
NVCCFLAGS    = -DUSE_VENDOR_BLAS  -O0 -G

NOOPTS       = -O0 -g
FORTRAN	     = /usr/local/bin/mpif77
FFLAGS	     =   

LOADER       = /usr/local/bin/mpicxx
LOADOPTS     =      
