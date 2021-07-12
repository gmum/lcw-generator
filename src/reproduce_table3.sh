# MNIST
python -OO -m train_generator --model cwg --dataset mnist  --monitor fid_score --noise_dim 32 --eval_fid ../data/mnist_fid_stats.npz --batch_size 256 --lr 0.0005
python -OO -m train_generator --model swg5000 --dataset mnist --monitor fid_score --noise_dim 32 --eval_fid ../data/mnist_fid_stats.npz --batch_size 256 --lr 0.0005
python -OO -m train_autoencoder --model cw2 --dataset mnist --monitor fid_score --latent_dim 8 --eval_fid ../data/mnist_fid_stats.npz --batch_size 128 --lr 0.001
python -OO -m train_latent_generator --model cwg --dataset mnist --noise_dim 8 --monitor fid_score --eval_fid ../data/mnist_fid_stats.npz --batch_size 256 --lr 0.0005  --ae_ckpt <provide path to checkpoint, i.e. ../results/ae/mnist/8/cw2/lightning_logs/version_0/checkpoints/epoch=199.ckpt>


# F-MNIST
python -OO -m train_generator --model cwg --dataset fmnist  --monitor fid_score --noise_dim 32 --eval_fid ../data/fmnist_fid_stats.npz --batch_size 256 --lr 0.0005
python -OO -m train_generator --model swg5000 --dataset fmnist --monitor fid_score --noise_dim 32 --eval_fid ../data/fmnist_fid_stats.npz --batch_size 256 --lr 0.0005
python -OO -m train_autoencoder --model cw2 --dataset fmnist --monitor fid_score --latent_dim 8 --eval_fid ../data/fmnist_fid_stats.npz --batch_size 128 --lr 0.001
python -OO -m train_latent_generator --model cwg --dataset fmnist --noise_dim 8 --monitor fid_score --eval_fid ../data/mnist_fid_stats.npz --batch_size 256 --lr 0.0005  --ae_ckpt <provide path to checkpoint, i.e. ../results/ae/fmnist/8/cw2/lightning_logs/version_0/checkpoints/epoch=199.ckpt>

# CELEBA
python -OO -m train_generator --model cwg --dataset celeba  --monitor fid_score --noise_dim 100 --eval_fid ../data/celeba_fid_stats.npz --batch_size 256 --lr 0.0005
python -OO -m train_generator --model swg5000 --dataset celeba --monitor fid_score --noise_dim 100 --eval_fid ../data/celeba_fid_stats.npz --batch_size 256 --lr 0.0005
python -OO -m train_autoencoder --model cw2 --lambda_val 0.2 --dataset celeba --monitor fid_score --latent_dim 128 --eval_fid ../data/celeba_fid_stats.npz --batch_size 128 --lr 0.0005
python -OO -m train_latent_generator --model cwg --dataset celeba --noise_dim 64 --monitor fid_score --eval_fid ../data/mnist_fid_stats.npz --batch_size 256 --lr 0.0005  --ae_ckpt <provide path to checkpoint, i.e. ../results/ae/celeba/128/cw2/lightning_logs/version_0/checkpoints/epoch=55.ckpt>

