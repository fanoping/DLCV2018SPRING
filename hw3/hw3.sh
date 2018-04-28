wget -O fcn32weights.20.hdf5 https://www.dropbox.com/s/x2jd17fuq8pf057/fcn32weights.20.hdf5?dl=1
python3 inference.py --input-dir $1 --output-dir $2 --file fcn32weights.20.hdf5
