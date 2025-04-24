numeric_file=pangulu_example.elf
Smatrix_name=$1
nb=$2
NP=$3
if [ ! -f $1 ];then
  echo "$1 is not a file."
  exit
fi

echo mpirun -np $NP ./$numeric_file -nb $nb -f $Smatrix_name

mpirun -np $NP ./$numeric_file -nb $nb -f $Smatrix_name
#mpirun --mca plm_rsh_args "-p 2007" -mca btl_tcp_if_include 192.168.2.1/24 -np $NP --machinefile hostfile \
#  ./$numeric_file -nb $nb -f $Smatrix_name
