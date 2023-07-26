# Set runtime env.
export LIBTORCH=$PWD/dependencies/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
ls ${LIBTORCH}/lib/libtorch_cpu.so || exit 1;
# Run
./obelisk_main