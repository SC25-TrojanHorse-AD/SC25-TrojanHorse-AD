ROOT_DIR=`pwd`

if [ -f "results_singlegpu" ]; then
    echo "results_singlegpu"
else
    ERRFLAG=1
    echo "Error: Directory 'results_singlegpu' not exist. Please get it by running 'run_single_gpu.sh' on RTX 4090 platform."
fi

if [ -f "results_singlegpu_4060" ]; then
    echo "results_singlegpu_4060"
else
    ERRFLAG=1
    echo "Error: Directory 'results_singlegpu_4060' not exist. Please get it by running 'run_single_gpu.sh' on RTX 4060 platform."
fi

if [ -f "results_multigpu" ]; then
    echo "results_multigpu"
else
    ERRFLAG=1
    echo "Error: Directory 'results_multigpu' not exist. Please get it by running 'run_multi_gpu.sh' on GPU cluster."
fi

if [ $ERRFLAG = "1" ]; then
    exit
fi


echo "Figure 8"
./scripts/004_figure8.sh $ROOT_DIR
cd $ROOT_DIR

echo "Figure 9"
./scripts/005_figure9.sh $ROOT_DIR
cd $ROOT_DIR

echo "Figure 10"
./scripts/007_figure10.sh $ROOT_DIR
cd $ROOT_DIR

echo "Figure 11"
./scripts/009_figure11.sh $ROOT_DIR
cd $ROOT_DIR

echo "Figure 12"
./scripts/011_figure12.sh $ROOT_DIR
cd $ROOT_DIR

cp Figures/Figure8/batch_compare_superlu.pdf Fig8a.pdf
cp Figures/Figure8/batch_compare_pangulu.pdf Fig8b.pdf

cp Figures/Figure9/SuperLU_scale_up.pdf Fig9a.pdf
cp Figures/Figure9/Pangu_scale_up.pdf Fig9b.pdf

cp Figures/Figure10/SuperLU_kernelCount.pdf Fig10a.pdf
cp Figures/Figure10/PanguLU_kernelCount.pdf Fig10b.pdf

cp Figures/Figure11/SuperLU_compare.pdf Fig11a.pdf
cp Figures/Figure11/PanguLU_compare.pdf Fig11b.pdf

cp Figures/Figure12/Scalability.pdf Fig12.pdf