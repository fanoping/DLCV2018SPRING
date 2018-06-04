wget -O rnn_checkpoint.pth.tar https://www.dropbox.com/s/a7rpo8ld3f7e9p6/rnn_epoch150_checkpoint.pth.tar?dl=1
python3 preprocess.py --mode trimmed --video-dir $1 --video-file trimmed_video.tar --csv-file $2
python3 rnn_inference.py --input-feature valid_feature.tar --input-video trimmed_video.tar --video-dir $1 --input-csv $2 --output-file $3 --checkpoint rnn_checkpoint.pth.tar --pretrained Resnet50
