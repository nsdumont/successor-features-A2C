python3 train.py --algo sr --model MiniGrid_SR_highlr --env MiniGrid-Empty-6x6-v0 --frames 100000 --seed 1 --curiosity True --lr 0.01
python3 train.py --algo sr --model MiniGrid_SR_midlr --env MiniGrid-Empty-6x6-v0 --frames 100000 --seed 1 --curiosity True --lr 0.001
python3 train.py --algo sr --model MiniGrid_SR_lowlr --env MiniGrid-Empty-6x6-v0 --frames 100000 --seed 1 --curiosity True --lr 0.0001

python3 train.py --algo sr --model MiniGrid_SR_recon --env MiniGrid-Empty-6x6-v0 --frames 100000 --seed 1 --curiosity False
python3 train.py --algo sr --model MiniGrid_SR_no_recon --env MiniGrid-Empty-6x6-v0 --frames 100000 --seed 1 --curiosity False --recon-loss-coef 0

python3 train.py --algo sr --model MiniGrid_SR_no_entropy --env MiniGrid-Empty-6x6-v0 --frames 100000 --seed 1 --curiosity True --entropy-coef 0
python3 train.py --algo sr --model MiniGrid_SR_high_entropy --env MiniGrid-Empty-6x6-v0 --frames 100000 --seed 1 --curiosity True --entropy-coef 0.1
python3 train.py --algo sr --model MiniGrid_SR_SRadvantage --env MiniGrid-Empty-6x6-v0 --frames 100000 --seed 1 --curiosity True --use-V-advantage False

python3 train.py --algo sr --model MiniGrid_SR_no_normloss --env MiniGrid-Empty-6x6-v0 --frames 100000 --seed 1 --curiosity True --norm-loss-coef 0
python3 train.py --algo sr --model MiniGrid_SR_no_rankloss --env MiniGrid-Empty-6x6-v0 --frames 100000 --seed 1 --curiosity True --rank-loss-coef 0

#norm loss rank loss



