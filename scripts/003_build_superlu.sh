ROOT_DIR=$1

cd SuperLU/SuperLU_benchmark/superlu_dist_9.1.0
rm -r ./build_gpu
./my_build_gpu.sh
cd build_gpu
make -j
cd $ROOT_DIR

cd SuperLU/SuperLU_benchmark/superlu_dist_sc25
rm -r ./build_gpu
./my_build_gpu.sh
cd build_gpu
make -j
cd $ROOT_DIR

cd SuperLU/SuperLU_kernel_count/superlu_dist_9.1.0_kernel_cnt_and_time
rm -r ./build_gpu
./my_build_gpu.sh
cd build_gpu
make -j
cd $ROOT_DIR

cd SuperLU/SuperLU_kernel_count/superlu_dist_sc25_kernel_cnt_and_time
rm -r ./build_gpu
./my_build_gpu.sh
cd build_gpu
make -j
cd $ROOT_DIR

cd SuperLU/SuperLU_kernel_gflops/superlu_dist_9.1.0_kernel_gflops
rm -r ./build_gpu
./my_build_gpu.sh
cd build_gpu
make -j
cd $ROOT_DIR

cd SuperLU/SuperLU_kernel_gflops/superlu_dist_sc25_kernel_gflops
rm -r ./build_gpu
./my_build_gpu.sh
cd build_gpu
make -j
cd $ROOT_DIR