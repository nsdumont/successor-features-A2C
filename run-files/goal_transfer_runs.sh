python3 train.py --algo a2c --model MiniGrid_Goal_A2C --env MiniGrid-EmptyGoal-8x8-v0 --frames 60000 --seed 1
python3 train.py --algo ppo --model MiniGrid_Goal_PPO --env MiniGrid-EmptyGoal-8x8-v0 --frames 60000 --seed 1
python3 train.py --algo sr --model MiniGrid_Goal_SR --env MiniGrid-EmptyGoal-8x8-v0 --frames 60000 --seed 1 --curiosity True --lr 0.001 --use-V-advantage False --recon-loss-coef 2

cp storage/MiniGrid_Goal_A2C/log.csv storage/MiniGrid_Goal_A2C/log1.csv
cp storage/MiniGrid_Goal_PPO/log.csv storage/MiniGrid_Goal_PPO/log1.csv
cp storage/MiniGrid_Goal_SR/log.csv storage/MiniGrid_Goal_SR/log1.csv

python3 train.py --algo a2c --model MiniGrid_Goal_A2C --env MiniGrid-EmptyGoal-8x8-v0 --frames 120000 --seed 5 --lr 0.0005
python3 train.py --algo ppo --model MiniGrid_Goal_PPO --env MiniGrid-EmptyGoal-8x8-v0 --frames 120000 --seed 5 --lr 0.0005
python3 train.py --algo sr --model MiniGrid_Goal_SR --env MiniGrid-EmptyGoal-8x8-v0 --frames 120000 --seed 5 --curiosity True --lr 0.0005 --use-V-advantage False --entropy-coef 0.007 --recon-loss-coef 2

cp storage/MiniGrid_Goal_A2C/log.csv storage/MiniGrid_Goal_A2C/log2.csv
cp storage/MiniGrid_Goal_PPO/log.csv storage/MiniGrid_Goal_PPO/log2.csv
cp storage/MiniGrid_Goal_SR/log.csv storage/MiniGrid_Goal_SR/log2.csv

python3 train.py --algo a2c --model MiniGrid_Goal_A2C --env MiniGrid-EmptyGoal-8x8-v0 --frames 180000 --seed 4 --lr 0.0005
python3 train.py --algo ppo --model MiniGrid_Goal_PPO --env MiniGrid-EmptyGoal-8x8-v0 --frames 180000 --seed 4 --lr 0.0005
python3 train.py --algo sr --model MiniGrid_Goal_SR --env MiniGrid-EmptyGoal-8x8-v0 --frames 180000 --seed 4 --curiosity True --lr 0.0005 --use-V-advantage False --recon-loss-coef 2 --entropy-coef 0.008

cp storage/MiniGrid_Goal_A2C/log.csv storage/MiniGrid_Goal_A2C/log3.csv
cp storage/MiniGrid_Goal_PPO/log.csv storage/MiniGrid_Goal_PPO/log3.csv
cp storage/MiniGrid_Goal_SR/log.csv storage/MiniGrid_Goal_SR/log3.csv



