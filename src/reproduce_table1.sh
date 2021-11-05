# MNIST

python -O -m train_generator --dataroot /shared/sets/datasets/vision/ --model cwg_dynamic --dataset mnist --monitor frechet_inception_distance --noise_dim 32 --batch_size 256 --lr 0.0005 --extra_tag experiment01 --gpus 1 --check_val_every_n_epoch 25 --progress_bar_refresh_rate 100 --patience 5
python -O -m train_generator --dataroot /shared/sets/datasets/vision/ --model cwg_fixed --dataset mnist --monitor frechet_inception_distance --noise_dim 32 --batch_size 256 --lr 0.0005 --extra_tag experiment01 --gpus 1 --check_val_every_n_epoch 25 --progress_bar_refresh_rate 100 --patience 5 --gamma 0.01160
python -O -m train_generator --dataroot /shared/sets/datasets/vision/ --model swg5000 --dataset mnist --monitor frechet_inception_distance --noise_dim 32 --batch_size 256 --lr 0.0005 --extra_tag experiment01 --gpus 1 --check_val_every_n_epoch 25 --progress_bar_refresh_rate 100 --patience 5

# F-MNIST
python -OO -m train_generator --model cwg --dataset fmnist  --monitor fid_score --noise_dim 32 --eval_fid ../data/fmnist_fid_stats.npz --batch_size 256 --lr 0.0005
python -OO -m train_generator --model swg5000 --dataset fmnist --monitor fid_score --noise_dim 32 --eval_fid ../data/fmnist_fid_stats.npz --batch_size 256 --lr 0.0005
python -OO -m train_autoencoder --model cwae --lambda_val 10 --dataset fmnist --monitor fid_score --latent_dim 8 --eval_fid ../data/fmnist_fid_stats.npz --batch_size 128 --lr 0.001
python -OO -m train_autoencoder --model cw2 --dataset fmnist --monitor fid_score --latent_dim 8 --eval_fid ../data/fmnist_fid_stats.npz --batch_size 128 --lr 0.001

# CELEBA
python -OO -m train_generator --model cwg --dataset celeba  --monitor fid_score --noise_dim 100 --eval_fid ../data/celeba_fid_stats.npz --batch_size 256 --lr 0.0005
python -OO -m train_generator --model swg5000 --dataset celeba --monitor fid_score --noise_dim 100 --eval_fid ../data/celeba_fid_stats.npz --batch_size 256 --lr 0.0005
python -OO -m train_autoencoder --model cwae --lambda_val 5 --dataset celeba --monitor fid_score --latent_dim 64 --eval_fid ../data/celeba_fid_stats.npz --batch_size 128 --lr 0.0005
python -OO -m train_autoencoder --model cw2 --lambda_val 0.2 --dataset celeba --monitor fid_score --latent_dim 64 --eval_fid ../data/celeba_fid_stats.npz --batch_size 128 --lr 0.0005

