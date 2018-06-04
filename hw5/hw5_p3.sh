wget -O seq2seq_best_checkpoint.pth.tar https://www.dropbox.com/s/0oeje5wo4z6orpd/seq2seq_best_checkpoint.pth.tar?dl=1
python3 preprocess.py --mode full-length --full-length-dir $1 --full-length-file full_length_video.tar
python3 seq2seq_inference.py --input-feature full_length_valid_feature.tar --full-length-file full_length_video.tar --full-length-dir $1 --output-file $2 --checkpoint seq2seq_best_checkpoint.pth.tar --pretrained Resnet50
