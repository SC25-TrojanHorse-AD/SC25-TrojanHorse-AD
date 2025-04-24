ROOT_DIR=$1

cd PanguLU/PanguLU_benchmark/PanguLU_4.2.0
make clean
make
make
cd $ROOT_DIR

cd PanguLU/PanguLU_benchmark/PanguLU_sc25
make clean
make
make
cd $ROOT_DIR

cd PanguLU/PanguLU_kernel_count/PanguLU_4.2.0_kernel_cnt_and_time
make clean
make
make
cd $ROOT_DIR

cd PanguLU/PanguLU_kernel_count/PanguLU_sc25_kernel_cnt_and_time
make clean
make
make
cd $ROOT_DIR

cd PanguLU/PanguLU_kernel_gflops/PanguLU_4.2.0_kernel_gflops
make clean
make
make
cd $ROOT_DIR

cd PanguLU/PanguLU_kernel_gflops/PanguLU_sc25_kernel_gflops
make clean
make
make
cd $ROOT_DIR