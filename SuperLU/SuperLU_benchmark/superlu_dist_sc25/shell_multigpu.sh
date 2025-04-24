ROOT_DIR=$1

mpirun -n 1 ./pddrive -r 1 -c 1 $ROOT_DIR/matrices/audikw_1/audikw_1.mtx
mpirun -n 2 ./pddrive -r 2 -c 1 $ROOT_DIR/matrices/audikw_1/audikw_1.mtx

mpirun -n 1 ./pddrive -r 1 -c 1 $ROOT_DIR/matrices/cage12/cage12.mtx
mpirun -n 2 ./pddrive -r 2 -c 1 $ROOT_DIR/matrices/cage12/cage12.mtx

mpirun -n 1 ./pddrive -r 1 -c 1 $ROOT_DIR/matrices/nlpkkt80/nlpkkt80.mtx
mpirun -n 2 ./pddrive -r 2 -c 1 $ROOT_DIR/matrices/nlpkkt80/nlpkkt80.mtx

mpirun -n 1 ./pddrive -r 1 -c 1 $ROOT_DIR/matrices/Si87H76/Si87H76.mtx
mpirun -n 2 ./pddrive -r 2 -c 1 $ROOT_DIR/matrices/Si87H76/Si87H76.mtx

mpirun -n 1 ./pddrive -r 1 -c 1 $ROOT_DIR/matrices/atmosmodd/atmosmodd.mtx
mpirun -n 2 ./pddrive -r 2 -c 1 $ROOT_DIR/matrices/atmosmodd/atmosmodd.mtx

mpirun -n 1 ./pddrive -r 1 -c 1 $ROOT_DIR/matrices/RM07R/RM07R.mtx
mpirun -n 2 ./pddrive -r 2 -c 1 $ROOT_DIR/matrices/RM07R/RM07R.mtx


