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
  LIBS = $(DFORTRANLIB) $(DSUPERLULIB) /opt/spack/opt/spack/linux-zen4/openblas-0.3.29-iwy6jdvpxol4335nmaelot64vt3nuhif/lib/libopenblas.a -lm
  LIBS +=  -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi -lgfortran -lm -lgcc_s -lgcc -lquadmath -lm -lc -lgcc_s -lgcc
else
  LIBS = $(DSUPERLULIB) /opt/spack/opt/spack/linux-zen4/openblas-0.3.29-iwy6jdvpxol4335nmaelot64vt3nuhif/lib/libopenblas.a -lm
endif

LIBS	 += 
LIBS	 += /opt/spack/opt/spack/linux-zen4/parmetis-4.0.3-sujmv5trzjpl7dfun2jaahhfmz63jmma/lib/libparmetis.so /opt/spack/opt/spack/linux-zen4/metis-5.1.0-w6kmssvl5a5n63gvhgnxnbpjcznhkuzw/lib/libmetis.so
LIBS	 += 
LIBS 	 += 
LIBS 	 += -lgcc_s -lgcc -lc -lgcc_s -lgcc -lmpi 
#LIBS     += CUDA::cudart CUDA::cublas CUDA::cusolver

CUDALIBS = /opt/spack/opt/spack/linux-zen4/cuda-12.8.1-v7u27lcd765atv2ijkto7qjrahbepg4r/lib64/libcudart.so /opt/spack/opt/spack/linux-zen4/cuda-12.8.1-v7u27lcd765atv2ijkto7qjrahbepg4r/lib64/libcublas.so /opt/spack/opt/spack/linux-zen4/cuda-12.8.1-v7u27lcd765atv2ijkto7qjrahbepg4r/lib64/libcusolver.so /opt/spack/opt/spack/linux-zen4/cuda-12.8.1-v7u27lcd765atv2ijkto7qjrahbepg4r/lib64/libcusparse.so
LIBS     += $(CUDALIBS)

#
#  The archiver and the flag(s) to use when building archive (library)

#  If your system has no ranlib, set RANLIB = echo.
#
ARCH         = /usr/bin/ar
ARCHFLAGS    = cr
RANLIB       = /usr/bin/ranlib

CC           = /opt/spack/opt/spack/linux-zen4/openmpi-5.0.7-nav6grha6p4y2gc6rq7xbm3fysgbj6ko/bin/mpicc
CFLAGS 	     = -O3 -DNDEBUG -I/opt/spack/opt/spack/linux-zen4/openmpi-5.0.7-nav6grha6p4y2gc6rq7xbm3fysgbj6ko/include -I/opt/spack/opt/spack/linux-zen4/parmetis-4.0.3-sujmv5trzjpl7dfun2jaahhfmz63jmma/include -I/opt/spack/opt/spack/linux-zen4/metis-5.1.0-w6kmssvl5a5n63gvhgnxnbpjcznhkuzw/include -I/opt/spack/opt/spack/linux-zen4/openblas-0.3.29-iwy6jdvpxol4335nmaelot64vt3nuhif/include -DUSE_VENDOR_BLAS -fopenmp   
##-fPIC
#CFLAGS      += -D
# CFLAGS     += 
CXX          = /opt/spack/opt/spack/linux-zen4/openmpi-5.0.7-nav6grha6p4y2gc6rq7xbm3fysgbj6ko/bin/mpicxx
CXXFLAGS     = -O3 -DNDEBUG -fopenmp  
NVCC	     = /opt/spack/opt/spack/linux-zen4/cuda-12.8.1-v7u27lcd765atv2ijkto7qjrahbepg4r/bin/nvcc
NVCCFLAGS    = -DUSE_VENDOR_BLAS  -O0 -G

NOOPTS       = -O0 -g
FORTRAN	     = /opt/spack/opt/spack/linux-zen4/openmpi-5.0.7-nav6grha6p4y2gc6rq7xbm3fysgbj6ko/bin/mpif77
FFLAGS	     =   

LOADER       = /opt/spack/opt/spack/linux-zen4/openmpi-5.0.7-nav6grha6p4y2gc6rq7xbm3fysgbj6ko/bin/mpicxx
LOADOPTS     =      
