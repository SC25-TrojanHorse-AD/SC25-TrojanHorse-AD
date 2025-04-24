ROOT_DIR=`pwd`

mkdir -p results_singlegpu

rm -r results_singlegpu/*

echo "1/12 Running PanguLU_4.2.0 benchmark"
cd PanguLU/PanguLU_benchmark/PanguLU_4.2.0/examples
rm nohup.out
nohup ./shell_singlegpu.sh $ROOT_DIR
cp nohup.out $ROOT_DIR/results_singlegpu/nohup_pangulu_420_benchmark.txt
cd $ROOT_DIR

echo "2/12 Running PanguLU+TrojanHorse benchmark"
cd PanguLU/PanguLU_benchmark/PanguLU_sc25/examples
rm nohup.out
nohup ./shell_singlegpu.sh $ROOT_DIR
cp nohup.out $ROOT_DIR/results_singlegpu/nohup_pangulu_sc25_benchmark.txt
cd $ROOT_DIR

echo "3/12 Running PanguLU_4.2.0 kernel breakdown"
cd PanguLU/PanguLU_kernel_count/PanguLU_4.2.0_kernel_cnt_and_time/examples
rm nohup.out
nohup ./shell_singlegpu.sh $ROOT_DIR
cp nohup.out $ROOT_DIR/results_singlegpu/nohup_pangulu_420_kernel_cnt_and_time.txt
cd $ROOT_DIR

echo "4/12 Running PanguLU+TrojanHorse kernel breakdown"
cd PanguLU/PanguLU_kernel_count/PanguLU_sc25_kernel_cnt_and_time/examples
rm nohup.out
nohup ./shell_singlegpu.sh $ROOT_DIR
cp nohup.out $ROOT_DIR/results_singlegpu/nohup_pangulu_sc25_kernel_cnt_and_time.txt
cd $ROOT_DIR

echo "5/12 Running PanguLU_4.2.0 kernel gflops"
cd PanguLU/PanguLU_kernel_gflops/PanguLU_4.2.0_kernel_gflops/examples
rm nohup.out
rm results/*
nohup ./shell_singlegpu.sh $ROOT_DIR
cp nohup.out $ROOT_DIR/results_singlegpu/nohup_pangulu_420_gflops.txt
rm -r $ROOT_DIR/results_singlegpu/kernel_gflops_pangulu_420
cp -r results $ROOT_DIR/results_singlegpu/kernel_gflops_pangulu_420
cd $ROOT_DIR

echo "6/12 Running PanguLU+TrojanHorse kernel gflops"
cd PanguLU/PanguLU_kernel_gflops/PanguLU_sc25_kernel_gflops/examples
rm nohup.out
rm results/*
nohup ./shell_singlegpu.sh $ROOT_DIR
cp nohup.out $ROOT_DIR/results_singlegpu/nohup_pangulu_sc25_gflops.txt
rm -r $ROOT_DIR/results_singlegpu/kernel_gflops_pangulu_sc25
cp -r results $ROOT_DIR/results_singlegpu/kernel_gflops_pangulu_sc25
cd $ROOT_DIR

echo "7/12 Running SuperLU_DIST_9.1.0 benchmark"
cd SuperLU/SuperLU_benchmark/superlu_dist_9.1.0/build_gpu/EXAMPLE
rm nohup.out
nohup ../../shell_singlegpu.sh $ROOT_DIR
cp nohup.out $ROOT_DIR/results_singlegpu/nohup_superlu_910_benchmark.txt
cd $ROOT_DIR

echo "8/12 Running SuperLU_DIST+TrojanHorse benchmark"
cd SuperLU/SuperLU_benchmark/superlu_dist_sc25/build_gpu/EXAMPLE
rm nohup.out
nohup ../../shell_singlegpu.sh $ROOT_DIR
cp nohup.out $ROOT_DIR/results_singlegpu/nohup_superlu_sc25_benchmark.txt
cd $ROOT_DIR

echo "9/12 Running SuperLU_DIST_9.1.0 kernel breakdown"
cd SuperLU/SuperLU_kernel_count/superlu_dist_9.1.0_kernel_cnt_and_time/build_gpu/EXAMPLE
rm nohup.out
nohup ../../shell_singlegpu.sh $ROOT_DIR
cp nohup.out $ROOT_DIR/results_singlegpu/nohup_superlu_910_kernel_cnt_and_time.txt
cd $ROOT_DIR

echo "10/12 Running SuperLU_DIST+TrojanHorse kernel breakdown"
cd SuperLU/SuperLU_kernel_count/superlu_dist_sc25_kernel_cnt_and_time/build_gpu/EXAMPLE
rm nohup.out
nohup ../../shell_singlegpu.sh $ROOT_DIR
cp nohup.out $ROOT_DIR/results_singlegpu/nohup_superlu_sc25_kernel_cnt_and_time.txt
cd $ROOT_DIR

echo "11/12 Running SuperLU_DIST_9.1.0 gflops"
cd SuperLU/SuperLU_kernel_gflops/superlu_dist_9.1.0_kernel_gflops/build_gpu/EXAMPLE
rm nohup.out
rm ./superlu_910_line_*.csv
nohup ../../shell_singlegpu.sh $ROOT_DIR
cp nohup.out $ROOT_DIR/results_singlegpu/nohup_superlu_910_kernel_gflops.txt
rm -r $ROOT_DIR/results_singlegpu/kernel_gflops_superlu_910
mkdir -p $ROOT_DIR/results_singlegpu/kernel_gflops_superlu_910
cp ./superlu_910_line_*.csv $ROOT_DIR/results_singlegpu/kernel_gflops_superlu_910
rm ./superlu_910_line_*.csv
cd $ROOT_DIR

echo "12/12 Running SuperLU_DIST+TrojanHorse gflops"
cd SuperLU/SuperLU_kernel_gflops/superlu_dist_sc25_kernel_gflops/build_gpu/EXAMPLE
rm nohup.out
rm ./superlu_sc25_line_*.csv
nohup ../../shell_singlegpu.sh $ROOT_DIR
cp nohup.out $ROOT_DIR/results_singlegpu/nohup_superlu_sc25_kernel_gflops.txt
rm -r $ROOT_DIR/results_singlegpu/kernel_gflops_superlu_sc25
mkdir -p $ROOT_DIR/results_singlegpu/kernel_gflops_superlu_sc25
cp ./superlu_sc25_line_*.csv $ROOT_DIR/results_singlegpu/kernel_gflops_superlu_sc25
rm ./superlu_sc25_line_*.csv
cd $ROOT_DIR
