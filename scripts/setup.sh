apt-get install -y wget zip unzip
wget -O libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
unzip -d dependencies libtorch.zip
rm libtorch.zip
# Weird problem fix.
cp dependencies/libtorch/lib/libgomp* dependencies/libtorch/lib/libgomp.so.1 || exit 1;
