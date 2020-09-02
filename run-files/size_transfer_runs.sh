python3 train.py --algo a2c --model MiniGrid_size_A2C --env MiniGrid-Empty-6x6-v0 --frames 50000 --seed 1
python3 train.py --algo ppo --model MiniGrid_size_PPO --env MiniGrid-Empty-6x6-v0 --frames 50000 --seed 1
python3 train.py --algo sr --model MiniGrid_size_SR --env MiniGrid-Empty-6x6-v0 --frames 50000 --seed 1 --curiosity True --lr 0.001 --use-V-advantage False --recon-loss-coef 2 

cp storage/MiniGrid_size_A2C/log.csv storage/MiniGrid_size_A2C/log1.csv
cp storage/MiniGrid_size_PPO/log.csv storage/MiniGrid_size_PPO/log1.csv
cp storage/MiniGrid_size_SR/log.csv storage/MiniGrid_size_SR/log1.csv

python3 train.py --algo a2c --model MiniGrid_size_A2C --env MiniGrid-Empty-8x8-v0 --frames 100000 --seed 1
python3 train.py --algo ppo --model MiniGrid_size_PPO --env MiniGrid-Empty-8x8-v0 --frames 100000 --seed 1
python3 train.py --algo sr --model MiniGrid_size_SR --env MiniGrid-Empty-8x8-v0 --frames 100000 --seed 1 --curiosity True --lr 0.001 --use-V-advantage False --recon-loss-coef 2 

cp storage/MiniGrid_size_A2C/log.csv storage/MiniGrid_size_A2C/log2.csv
cp storage/MiniGrid_size_PPO/log.csv storage/MiniGrid_size_PPO/log2.csv
cp storage/MiniGrid_size_SR/log.csv storage/MiniGrid_size_SR/log2.csv

python3 train.py --algo a2c --model MiniGrid_size_A2C --env MiniGrid-Empty-16x16-v0 --frames 150000 --seed 1
python3 train.py --algo ppo --model MiniGrid_size_PPO --env MiniGrid-Empty-16x16-v0 --frames 150000 --seed 1
python3 train.py --algo sr --model MiniGrid_size_SR --env MiniGrid-Empty-16x16-v0 --frames 150000 --seed 1 --curiosity True --lr 0.001 --use-V-advantage False --entropy-coef 0.007 --recon-loss-coef 2

cp storage/MiniGrid_size_A2C/log.csv storage/MiniGrid_size_A2C/log3.csv
cp storage/MiniGrid_size_PPO/log.csv storage/MiniGrid_size_PPO/log3.csv
cp storage/MiniGrid_size_SR/log.csv storage/MiniGrid_size_SR/log3.csv
