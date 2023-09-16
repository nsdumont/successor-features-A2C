python train.py --algo a2c --model MiniGrid_NoGoal_A2C_image --env MiniGrid-EmptyNoGoal-Random-8x8-v0 --frames 10000 --input image --seed 1 --entropy-coef 0.0005 --load-optimizer-state False
python train.py --algo ppo --model MiniGrid_NoGoal_PPO_image --env MiniGrid-EmptyNoGoal-Random-8x8-v0 --frames 10000 --input image --seed 1 --entropy-coef 0.0005 --load-optimizer-state False
python train.py --algo sr --model MiniGrid_NoGoal_SR_image --env MiniGrid-EmptyNoGoal-Random-8x8-v0 --frames 10000 --input image --feature-learn curiosity --lr_r 0 --entropy-coef 0.01 --seed 1 --load-optimizer-state False

python train.py --algo a2c --model MiniGrid_NoGoal_A2C_ssp --env MiniGrid-EmptyNoGoal-Random-8x8-v0 --frames 10000 --input ssp-xy  --seed 1 --entropy-coef 0.0005
python train.py --algo ppo --model MiniGrid_NoGoal_PPO_ssp --env MiniGrid-EmptyNoGoal-Random-8x8-v0 --frames 10000 --input ssp-xy  --seed 1 --entropy-coef 0.0005
python train.py --algo sr --model MiniGrid_NoGoal_SR_ssp --env MiniGrid-EmptyNoGoal-Random-8x8-v0 --frames 10000 --input ssp-xy --feature-learn none  --lr_r 0 --entropy-coef 0.01  --seed 1



python train.py --algo a2c --model MiniGrid_NoGoal_A2C_image --env MiniGrid-Empty-Random-8x8-v0 --frames 50000 --input image  --seed 1  --entropy-coef 0.0005 --load-optimizer-state True
python train.py --algo ppo --model MiniGrid_NoGoal_PPO_image --env MiniGrid-Empty-Random-8x8-v0 --frames 50000 --input image  --seed 1 --entropy-coef 0.0005 --load-optimizer-state True
python train.py --algo sr --model MiniGrid_NoGoal_SR_image --env MiniGrid-Empty-Random-8x8-v0 --frames 50000 --input image --feature-learn curiosity  --entropy-coef 0.0005  --load-optimizer-state True  --seed 1

python train.py --algo a2c --model MiniGrid_NoGoal_A2C_ssp --env MiniGrid-Empty-Random-8x8-v0 --frames 50000 --input ssp-xy --seed 1 --entropy-coef 0.0005 --load-optimizer-state True
python train.py --algo ppo --model MiniGrid_NoGoal_PPO_ssp --env MiniGrid-Empty-Random-8x8-v0 --frames 50000 --input ssp-xy --seed 1 --entropy-coef 0.0005 --load-optimizer-state True
python train.py --algo sr --model MiniGrid_NoGoal_SR_ssp --env MiniGrid-Empty-Random-8x8-v0 --frames 50000 --input ssp-xy --feature-learn none  --entropy-coef 0.0005  --load-optimizer-state True  --seed 1

