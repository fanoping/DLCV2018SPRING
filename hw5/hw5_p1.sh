wget -O cnn_checkpoint.pth.tar https://www.dropbox.com/s/579r8tg4dwvtcmp/cnn_epoch150_checkpoint.pth.tar?dl=1
python3 preprocess.py --mode trimmed --video-dir $1 --video-file cnn_trimmed_video.tar --csv-file $2
python3 cnn_inference.py --input-feature cnn_valid_feature.tar --input-video cnn_trimmed_video.tar --video-dir $1 --input-csv $2 --output-file $3 --checkpoint cnn_checkpoint.pth.tar --pretrained Resnet50
