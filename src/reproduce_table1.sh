# MNIST
python -OO -m train_generator --model cwg --dataset mnist  --monitor fid_score --noise_dim 32 --eval_fid ../data/mnist_fid_stats.npz --batch_size 256 --lr 0.0005
python -OO -m train_generator --model swg5000 --dataset mnist --monitor fid_score --noise_dim 32 --eval_fid ../data/mnist_fid_stats.npz --batch_size 256 --lr 0.0005
python -OO -m train_autoencoder --model cwae --dataset mnist --monitor fid_score --latent_dim 8 --eval_fid ../data/mnist_fid_stats.npz --batch_size 128 --lr 0.001
python -OO -m train_autoencoder --model cw2 --dataset mnist --monitor fid_score --latent_dim 8 --eval_fid ../data/mnist_fid_stats.npz --batch_size 128 --lr 0.001

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

