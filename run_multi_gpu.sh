ROOT_DIR=`pwd`

mkdir -p results_multigpu

rm -r results_multigpu/*

echo "1/4 Running PanguLU_4.2.0 benchmark"
cd PanguLU/PanguLU_benchmark/PanguLU_4.2.0/examples
rm nohup.out
nohup ./shell_multigpu.sh $ROOT_DIR
cp nohup.out $ROOT_DIR/results_multigpu/nohup_pangulu_420_benchmark.txt
cd $ROOT_DIR

echo "2/4 Running PanguLU+TrojanHorse benchmark"
cd PanguLU/PanguLU_benchmark/PanguLU_sc25/examples
rm nohup.out
nohup ./shell_multigpu.sh $ROOT_DIR
cp nohup.out $ROOT_DIR/results_multigpu/nohup_pangulu_sc25_benchmark.txt
cd $ROOT_DIR

echo "3/4 Running SuperLU_DIST_9.1.0 benchmark"
cd SuperLU/SuperLU_benchmark/superlu_dist_9.1.0/build_gpu/EXAMPLE
rm nohup.out
nohup ../../shell_multigpu.sh $ROOT_DIR
cp nohup.out $ROOT_DIR/results_multigpu/nohup_superlu_910_benchmark.txt
cd $ROOT_DIR

echo "4/4 Running SuperLU_DIST+TrojanHorse benchmark"
cd SuperLU/SuperLU_benchmark/superlu_dist_sc25/build_gpu/EXAMPLE
rm nohup.out
nohup ../../shell_multigpu.sh $ROOT_DIR
cp nohup.out $ROOT_DIR/results_multigpu/nohup_superlu_sc25_benchmark.txt
cd $ROOT_DIR