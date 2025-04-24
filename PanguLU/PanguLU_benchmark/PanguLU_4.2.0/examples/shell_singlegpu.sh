ROOT_DIR=$1

mpirun -n 1 ./pangulu_example.elf -nb 500 -f $ROOT_DIR/matrices/ex11/ex11.mtx
mpirun -n 1 ./pangulu_example.elf -nb 500 -f $ROOT_DIR/matrices/gas_sensor/gas_sensor.mtx
mpirun -n 1 ./pangulu_example.elf -nb 500 -f $ROOT_DIR/matrices/shipsec1/shipsec1.mtx
mpirun -n 1 ./pangulu_example.elf -nb 500 -f $ROOT_DIR/matrices/para-8/para-8.mtx
mpirun -n 1 ./pangulu_example.elf -nb 500 -f $ROOT_DIR/matrices/inline_1/inline_1.mtx
mpirun -n 1 ./pangulu_example.elf -nb 500 -f $ROOT_DIR/matrices/ldoor/ldoor.mtx
