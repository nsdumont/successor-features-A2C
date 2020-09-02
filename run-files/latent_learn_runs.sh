python3 train.py --algo a2c --model MiniGrid_NoGoal_A2C --env MiniGrid-EmptyNoGoal-8x8-v0 --frames 10000 --seed 1
python3 train.py --algo ppo --model MiniGrid_NoGoal_PPO --env MiniGrid-EmptyNoGoal-8x8-v0 --frames 10000 --seed 1
python3 train.py --algo sr --model MiniGrid_NoGoal_SR --env MiniGrid-EmptyNoGoal-8x8-v0 --frames 10000 --seed 1 --curiosity True --lr 0.001 --use-V-advantage False --recon-loss-coef 5 -entropy-coef 0.1

cp storage/MiniGrid_NoGoal_A2C/log.csv storage/MiniGrid_NoGoal_A2C/log1.csv
cp storage/MiniGrid_NoGoal_PPO/log.csv storage/MiniGrid_NoGoal_PPO/log1.csv
cp storage/MiniGrid_NoGoal_SR/log.csv storage/MiniGrid_NoGoal_SR/log1.csv

python3 train.py --algo a2c --model MiniGrid_NoGoal_A2C --env MiniGrid-Empty-8x8-v0 --frames 100000 --seed 5 --lr 0.0005 --load-optimizer-state False
python3 train.py --algo ppo --model MiniGrid_NoGoal_PPO --env MiniGrid-Empty-8x8-v0 --frames 100000 --seed 5 --lr 0.0005 --load-optimizer-state False
python3 train.py --algo sr --model MiniGrid_NoGoal_SR --env MiniGrid-Empty-8x8-v0 --frames 100000 --seed 5 --curiosity True --lr 0.001 --use-V-advantage False --entropy-coef 0.005 --recon-loss-coef 2 --load-optimizer-state False

cp storage/MiniGrid_NoGoal_A2C/log.csv storage/MiniGrid_NoGoal_A2C/log2.csv
cp storage/MiniGrid_NoGoal_PPO/log.csv storage/MiniGrid_NoGoal_PPO/log2.csv
cp storage/MiniGrid_NoGoal_SR/log.csv storage/MiniGrid_NoGoal_SR/log2.csv
