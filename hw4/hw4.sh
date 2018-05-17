wget -O vae_best_checkpoint.pth.tar https://www.dropbox.com/s/aqwfaq2ofzgkuro/vae_best_checkpoint.pth.tar?dl=1
python3 vae_inference.py --input-file $1 --output-file $2 --checkpoint vae_best_checkpoint.pth.tar
wget -O gan_epoch230_checkpoint.pth.tar https://www.dropbox.com/s/t7jnvbj0oawnhdi/gan_epoch230_checkpoint.pth.tar?dl=1
python3 gan_inference.py --output-file $2 --checkpoint gan_epoch230_checkpoint.pth.tar
wget -O acgan_epoch300_checkpoint.pth.tar https://www.dropbox.com/s/1017nvp3rnsu5x0/acgan_epoch300_checkpoint.pth.tar?dl=1
python3 acgan_inference.py --output-file $2 --checkpoint acgan_epoch300_checkpoint.pth.tar
wget -O infogan_epoch86_checkpoint.pth.tar https://www.dropbox.com/s/3whz6sk5ahnedf8/infogan_epoch86_checkpoint.pth.tar?dl=1
python3 infogan_inference.py --output-file $2 --checkpoint infogan_epoch86_checkpoint.pth.tar
