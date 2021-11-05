# MNIST
python -O -m train_autoencoder --model ae --dataset mnist --latent_dim 8 --batch_size 128 --lr 0.001 --gpus 1 --save_checkpoint --check_val_every_n_epoch 20 --patience 5
python -O -m train_autoencoder --model cwae --dataset mnist --latent_dim 8 --batch_size 128 --lr 0.001 --gpus 1 --save_checkpoint --check_val_every_n_epoch 20 --patience 5
python -O -m train_autoencoder --model cw2_fixed --dataset mnist --latent_dim 8 --batch_size 256 --lr 0.002 --gamma_val 0.0116 --gpus 1 --save_checkpoint --check_val_every_n_epoch 20 --patience 5
python -O -m train_latent_generator --model cwg_fixed --dataset mnist --noise_dim 8  --batch_size 256 --lr 0.002 --gpus 1 --check_val_every_n_epoch 20 --patience 5 --ae_ckpt <ae_or_cw2_checkpoint_path>

# F-MNIST
python -O -m train_autoencoder --model ae --dataset fmnist --latent_dim 8 --batch_size 128 --lr 0.001 --gpus 1 --save_checkpoint --check_val_every_n_epoch 20 --patience 5
python -O -m train_autoencoder --model cwae --dataset fmnist --lambda_val 10 --latent_dim 8 --batch_size 128 --lr 0.001 --gpus 1 --save_checkpoint --check_val_every_n_epoch 20 --patience 5
python -O -m train_autoencoder --model cw2_fixed --dataset fmnist --latent_dim 8 --batch_size 256 --lr 0.002 --gamma_val 0.0152 --gpus 1 --save_checkpoint --check_val_every_n_epoch 20 --patience 5
python -O -m train_latent_generator --model cwg_fixed --dataset fmnist --noise_dim 8 --batch_size 256 --lr 0.002 --gpus 1 --check_val_every_n_epoch 20 --patience 5 --ae_ckpt <ae_or_cw2_checkpoint_path>

# KMNIST
python -O -m train_autoencoder --model ae --dataset kmnist --latent_dim 8 --batch_size 128 --lr 0.001 --gpus 1 --save_checkpoint --check_val_every_n_epoch 20 --patience 5
python -O -m train_autoencoder --model cwae --dataset kmnist --lambda_val 10 --latent_dim 8 --batch_size 128 --lr 0.001 --gpus 1 --save_checkpoint --check_val_every_n_epoch 20 --patience 5
python -O -m train_autoencoder --model cw2_fixed --dataset kmnist --latent_dim 8 --batch_size 256 --lr 0.002 --gamma_val 0.0148 --gpus 1 --save_checkpoint --check_val_every_n_epoch 20 --patience 5
python -O -m train_latent_generator --model cwg_fixed --dataset kmnist --noise_dim 8 --batch_size 256 --lr 0.002 --gpus 1 --check_val_every_n_epoch 20 --patience 5 --ae_ckpt <ae_or_cw2_checkpoint_path>

# SVHN
python -O -m train_autoencoder --model ae --dataset svhn --latent_dim 48 --batch_size 128 --lr 0.001 --gpus 1 --save_checkpoint --check_val_every_n_epoch 20 --patience 5
python -O -m train_autoencoder --model cwae --dataset svhn --lambda_val 10 --latent_dim 48 --batch_size 128 --lr 0.001 --gpus 1 --save_checkpoint --check_val_every_n_epoch 20 --patience 5
python -O -m train_autoencoder --model cw2_fixed --dataset svhn --latent_dim 48 --batch_size 256 --lr 0.002 --gamma_val 0.0049 --gpus 1 --save_checkpoint --check_val_every_n_epoch 20 --patience 5
python -O -m train_latent_generator --model cwg_fixed --dataset svhn --noise_dim 48 --batch_size 256 --lr 0.002 --gpus 1 --check_val_every_n_epoch 20 --patience 5 --ae_ckpt <ae_or_cw2_checkpoint_path>

# CELEBA
python -O -m train_autoencoder --model ae --dataset celeba  --latent_dim 64 --batch_size 128 --lr 0.001 --gpus 1 --save_checkpoint --check_val_every_n_epoch 5 --patience 5
python -O -m train_autoencoder --model cwae --dataset celeba --lambda_val 10 --latent_dim 64 --batch_size 128 --lr 0.001 --gpus 1 --save_checkpoint --check_val_every_n_epoch 5 --patience 5
python -O -m train_autoencoder --model cw2_fixed --dataset celeba --lambda_val 0.2 --latent_dim 64 --batch_size 256 --lr 0.002 --gamma_val 0.0094 --gpus 1 --check_val_every_n_epoch 5 --patience 5

## For LCW
python -O -m train_autoencoder --model ae --dataset celeba --latent_dim 128 --batch_size 128 --lr 0.001 --gpus 1 --save_checkpoint --check_val_every_n_epoch 5 --patience 5
python -O -m train_autoencoder --model cw2_fixed --dataset celeba --lambda_val 0.2 --latent_dim 128 --batch_size 256 --lr 0.002 --gamma_val 0.0094 --gpus 1 --save_checkpoint --check_val_every_n_epoch 5 --patience 5
python -O -m train_latent_generator --model cwg_fixed --dataset celeba --noise_dim 32 --batch_size 256 --lr 0.002 --gpus 1 --check_val_every_n_epoch 5 --patience 5 --ae_ckpt <ae_or_cw2_checkpoint_path>