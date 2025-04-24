mkdir -p matrices
cd matrices

PREFIX=http://sparse-files.engr.tamu.edu

wget $PREFIX/MM/GHS_psdef/ldoor.tar.gz # 240MB, 3min
wget $PREFIX/MM/GHS_psdef/inline_1.tar.gz # 207MB, 3min
wget $PREFIX/MM/Schenk_ISEI/para-8.tar.gz # 26MB, 1min
wget $PREFIX/MM/Oberwolfach/gas_sensor.tar.gz # 3.4MB, 5s
wget $PREFIX/MM/DNVS/shipsec1.tar.gz # 6.7MB, 5s
wget $PREFIX/MM/FIDAP/ex11.tar.gz # 3.6MB, 5s
wget $PREFIX/MM/GHS_psdef/audikw_1.tar.gz # 418MB, 8min
wget $PREFIX/MM/vanHeukelum/cage12.tar.gz # 8.5MB, 10s
wget $PREFIX/MM/Schenk/nlpkkt80.tar.gz # 45MB, 1min
wget $PREFIX/MM/PARSEC/Si87H76.tar.gz # 33MB, 1min
wget $PREFIX/MM/Bourchtein/atmosmodd.tar.gz # 47MB, 1min
wget $PREFIX/MM/Fluorem/RM07R.tar.gz # 423MB, 8min

# 3min
tar -xf ./ldoor.tar.gz
tar -xf ./inline_1.tar.gz
tar -xf ./para-8.tar.gz
tar -xf ./gas_sensor.tar.gz
tar -xf ./shipsec1.tar.gz
tar -xf ./ex11.tar.gz
tar -xf ./audikw_1.tar.gz
tar -xf ./cage12.tar.gz
tar -xf ./nlpkkt80.tar.gz
tar -xf ./Si87H76.tar.gz
tar -xf ./atmosmodd.tar.gz
tar -xf ./RM07R.tar.gz