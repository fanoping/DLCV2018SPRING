wget -O fcn16weights.20.hdf5 https://www.dropbox.com/s/sf6kjr36dujmsdx/fcn16weights.20.hdf5?dl=1
python3 inference.py --input-dir $1 --output-dir $2 --file fcn16weights.20.hdf5
