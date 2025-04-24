ROOT_DIR=$1

cp $ROOT_DIR/results_singlegpu/kernel_gflops_pangulu_420/line_ex11.csv $ROOT_DIR/Figures/Figure8/ex11/4090_line_ex11_nobatch.csv
cp $ROOT_DIR/results_singlegpu/kernel_gflops_pangulu_420/line_gas_sensor.csv $ROOT_DIR/Figures/Figure8/gas_sensor/4090_line_gas_sensor_nobatch.csv
cp $ROOT_DIR/results_singlegpu/kernel_gflops_pangulu_420/line_inline_1.csv $ROOT_DIR/Figures/Figure8/inline_1/4090_line_inline_1_nobatch.csv
cp $ROOT_DIR/results_singlegpu/kernel_gflops_pangulu_420/line_ldoor.csv $ROOT_DIR/Figures/Figure8/ldoor/4090_line_ldoor_nobatch.csv
cp $ROOT_DIR/results_singlegpu/kernel_gflops_pangulu_420/line_para-8.csv $ROOT_DIR/Figures/Figure8/para-8/4090_line_para-8_nobatch.csv
cp $ROOT_DIR/results_singlegpu/kernel_gflops_pangulu_420/line_shipsec1.csv $ROOT_DIR/Figures/Figure8/shipsec1/4090_line_shipsec1_nobatch.csv

cp $ROOT_DIR/results_singlegpu/kernel_gflops_pangulu_sc25/line_ex11.csv $ROOT_DIR/Figures/Figure8/ex11/4090_line_ex11_batch.csv
cp $ROOT_DIR/results_singlegpu/kernel_gflops_pangulu_sc25/line_gas_sensor.csv $ROOT_DIR/Figures/Figure8/gas_sensor/4090_line_gas_sensor_batch.csv
cp $ROOT_DIR/results_singlegpu/kernel_gflops_pangulu_sc25/line_inline_1.csv $ROOT_DIR/Figures/Figure8/inline_1/4090_line_inline_1_batch.csv
cp $ROOT_DIR/results_singlegpu/kernel_gflops_pangulu_sc25/line_ldoor.csv $ROOT_DIR/Figures/Figure8/ldoor/4090_line_ldoor_batch.csv
cp $ROOT_DIR/results_singlegpu/kernel_gflops_pangulu_sc25/line_para-8.csv $ROOT_DIR/Figures/Figure8/para-8/4090_line_para-8_batch.csv
cp $ROOT_DIR/results_singlegpu/kernel_gflops_pangulu_sc25/line_shipsec1.csv $ROOT_DIR/Figures/Figure8/shipsec1/4090_line_shipsec1_batch.csv

cp $ROOT_DIR/results_singlegpu/kernel_gflops_superlu_910/superlu_910_line_ex11.csv $ROOT_DIR/Figures/Figure8/superlu_ex11/superlu_line_ex11_nobatch.csv
cp $ROOT_DIR/results_singlegpu/kernel_gflops_superlu_910/superlu_910_line_gas_sensor.csv $ROOT_DIR/Figures/Figure8/superlu_gas_sensor/superlu_line_gas_sensor_nobatch.csv
cp $ROOT_DIR/results_singlegpu/kernel_gflops_superlu_910/superlu_910_line_inline_1.csv $ROOT_DIR/Figures/Figure8/superlu_inline_1/superlu_line_inline_1_nobatch.csv
cp $ROOT_DIR/results_singlegpu/kernel_gflops_superlu_910/superlu_910_line_ldoor.csv $ROOT_DIR/Figures/Figure8/superlu_ldoor/superlu_line_ldoor_nobatch.csv
cp $ROOT_DIR/results_singlegpu/kernel_gflops_superlu_910/superlu_910_line_para-8.csv $ROOT_DIR/Figures/Figure8/superlu_para-8/superlu_line_para-8_nobatch.csv
cp $ROOT_DIR/results_singlegpu/kernel_gflops_superlu_910/superlu_910_line_shipsec1.csv $ROOT_DIR/Figures/Figure8/superlu_shipsec1/superlu_line_shipsec1_nobatch.csv

cp $ROOT_DIR/results_singlegpu/kernel_gflops_superlu_sc25/superlu_sc25_line_ex11.csv $ROOT_DIR/Figures/Figure8/superlu_ex11/superlu_line_ex11_batch.csv
cp $ROOT_DIR/results_singlegpu/kernel_gflops_superlu_sc25/superlu_sc25_line_gas_sensor.csv $ROOT_DIR/Figures/Figure8/superlu_gas_sensor/superlu_line_gas_sensor_batch.csv
cp $ROOT_DIR/results_singlegpu/kernel_gflops_superlu_sc25/superlu_sc25_line_inline_1.csv $ROOT_DIR/Figures/Figure8/superlu_inline_1/superlu_line_inline_1_batch.csv
cp $ROOT_DIR/results_singlegpu/kernel_gflops_superlu_sc25/superlu_sc25_line_ldoor.csv $ROOT_DIR/Figures/Figure8/superlu_ldoor/superlu_line_ldoor_batch.csv
cp $ROOT_DIR/results_singlegpu/kernel_gflops_superlu_sc25/superlu_sc25_line_para-8.csv $ROOT_DIR/Figures/Figure8/superlu_para-8/superlu_line_para-8_batch.csv
cp $ROOT_DIR/results_singlegpu/kernel_gflops_superlu_sc25/superlu_sc25_line_shipsec1.csv $ROOT_DIR/Figures/Figure8/superlu_shipsec1/superlu_line_shipsec1_batch.csv

cd $ROOT_DIR/Figures/Figure8
./00run.sh