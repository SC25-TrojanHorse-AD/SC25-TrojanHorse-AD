ROOT_DIR=`pwd`

rm -rf ./results*
rm -rf matrices

cd PanguLU/PanguLU_benchmark/PanguLU_4.2.0
make clean
rm examples/nohup.out
cd $ROOT_DIR

cd PanguLU/PanguLU_benchmark/PanguLU_sc25
make clean
rm examples/nohup.out
cd $ROOT_DIR

cd PanguLU/PanguLU_kernel_count/PanguLU_4.2.0_kernel_cnt_and_time
make clean
rm examples/nohup.out
cd $ROOT_DIR

cd PanguLU/PanguLU_kernel_count/PanguLU_sc25_kernel_cnt_and_time
make clean
rm examples/nohup.out
cd $ROOT_DIR

cd PanguLU/PanguLU_kernel_gflops/PanguLU_4.2.0_kernel_gflops
make clean
rm examples/nohup.out
rm examples/results/*
cd $ROOT_DIR

cd PanguLU/PanguLU_kernel_gflops/PanguLU_sc25_kernel_gflops
make clean
rm examples/nohup.out
rm examples/results/*
cd $ROOT_DIR

cd SuperLU/SuperLU_benchmark/superlu_dist_9.1.0
rm -rf build_gpu
cd $ROOT_DIR

cd SuperLU/SuperLU_benchmark/superlu_dist_sc25
rm -rf build_gpu
cd $ROOT_DIR

cd SuperLU/SuperLU_kernel_count/superlu_dist_9.1.0_kernel_cnt_and_time
rm -rf build_gpu
cd $ROOT_DIR

cd SuperLU/SuperLU_kernel_count/superlu_dist_sc25_kernel_cnt_and_time
rm -rf build_gpu
cd $ROOT_DIR

cd SuperLU/SuperLU_kernel_gflops/superlu_dist_9.1.0_kernel_gflops
rm -rf build_gpu
cd $ROOT_DIR

cd SuperLU/SuperLU_kernel_gflops/superlu_dist_sc25_kernel_gflops
rm -rf build_gpu
cd $ROOT_DIR

cd Figures/Figure8
./clean.sh
cd $ROOT_DIR

cd Figures/Figure9
./clean.sh
cd $ROOT_DIR

cd Figures/Figure10
./clean.sh
cd $ROOT_DIR

cd Figures/Figure11
./clean.sh
cd $ROOT_DIR

cd Figures/Figure12
./clean.sh
cd $ROOT_DIR

rm ./*.pdf