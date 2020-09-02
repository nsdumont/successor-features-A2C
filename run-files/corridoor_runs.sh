python3 train.py --algo a2c --model MiniGrid_corridoor_A2C --env MiniGrid-FourCorridors1-v0 --frames 100000 --seed 1
python3 train.py --algo ppo --model MiniGrid_corridoor_PPO --env MiniGrid-FourCorridors1-v0 --frames 100000 --seed 1
python3 train.py --algo sr --model MiniGrid_corridoor_SR --env MiniGrid-FourCorridors1-v0 --frames 100000 --seed 1 --curiosity True --lr 0.001 --use-V-advantage False

cp storage/MiniGrid_corridoor_A2C/log.csv storage/MiniGrid_corridoor_A2C/log1.csv
cp storage/MiniGrid_corridoor_PPO/log.csv storage/MiniGrid_corridoor_PPO/log1.csv
cp storage/MiniGrid_corridoor_SR/log.csv storage/MiniGrid_corridoor_SR/log1.csv

python3 train.py --algo a2c --model MiniGrid_corridoor_A2C --env MiniGrid-FourCorridors2-v0 --frames 200000 --seed 1
python3 train.py --algo ppo --model MiniGrid_corridoor_PPO --env MiniGrid-FourCorridors2-v0 --frames 200000 --seed 1
python3 train.py --algo sr --model MiniGrid_corridoor_SR --env MiniGrid-FourCorridors2-v0 --frames 200000 --seed 1 --curiosity True --lr 0.001 --use-V-advantage False

cp storage/MiniGrid_corridoor_A2C/log.csv storage/MiniGrid_corridoor_A2C/log2.csv
cp storage/MiniGrid_corridoor_PPO/log.csv storage/MiniGrid_corridoor_PPO/log2.csv
cp storage/MiniGrid_Goal_SR/log.csv storage/MiniGrid_Goal_SR/log2.csv

python3 train.py --algo a2c --model MiniGrid_corridoor_A2C --env MiniGrid-FourCorridors3-v0 --frames 300000 --seed 1
python3 train.py --algo ppo --model MiniGrid_corridoor_PPO --env MiniGrid-FourCorridors3-v0 --frames 300000 --seed 1
python3 train.py --algo sr --model MiniGrid_corridoor_SR --env MiniGrid-FourCorridors3-v0 --frames 300000 --seed 1 --curiosity True --lr 0.001 --use-V-advantage False --entropy-coef 0.01

cp storage/MiniGrid_corridoor_A2C/log.csv storage/MiniGrid_corridoor_A2C/log3.csv
cp storage/MiniGrid_corridoor_PPO/log.csv storage/MiniGrid_corridoor_PPO/log3.csv
cp storage/MiniGrid_Goal_SR/log.csv storage/MiniGrid_Goal_SR/log3.csv

python3 train.py --algo a2c --model MiniGrid_corridoor_A2C --env MiniGrid-FourCorridors4-v0 --frames 400000 --seed 1
python3 train.py --algo ppo --model MiniGrid_corridoor_PPO --env MiniGrid-FourCorridors4-v0 --frames 400000 --seed 1
python3 train.py --algo sr --model MiniGrid_corridoor_SR --env MiniGrid-FourCorridors4-v0 --frames 400000 --seed 1 --curiosity True --lr 0.001 --use-V-advantage False --entropy-coef 0.01

cp storage/MiniGrid_corridoor_A2C/log.csv storage/MiniGrid_corridoor_A2C/log4.csv
cp storage/MiniGrid_corridoor_PPO/log.csv storage/MiniGrid_corridoor_PPO/log4.csv
cp storage/MiniGrid_Goal_SR/log.csv storage/MiniGrid_Goal_SR/log4.csv
