python train.py --algo a2c --model MiniGrid_size_A2C_image --env MiniGrid-Empty-6x6-v0 --frames 40000 --input image --load-optimizer-state False
python train.py --algo ppo --model MiniGrid_size_PPO_image --env MiniGrid-Empty-6x6-v0 --frames 40000 --input image  --load-optimizer-state False
python train.py --algo sr --model MiniGrid_size_SR_image --env MiniGrid-Empty-6x6-v0 --frames 40000 --input image --feature-learn curiosity --seed 1 --entropy-coef 0.01 --entropy-decay 0.9 --load-optimizer-state False

python train.py --algo a2c --model MiniGrid_size_A2C_ssp --env MiniGrid-Empty-6x6-v0 --frames 40000 --input ssp-xy  --seed 1 --load-optimizer-state False
python train.py --algo ppo --model MiniGrid_size_PPO_ssp --env MiniGrid-Empty-6x6-v0 --frames 40000 --input ssp-xy  --seed 1 --load-optimizer-state False
python train.py --algo sr --model MiniGrid_size_SR_ssp --env MiniGrid-Empty-6x6-v0 --frames 40000 --input ssp-xy --feature-learn none  --entropy-coef 0.01 --entropy-decay 0.9  --seed 1 --load-optimizer-state False


python train.py --algo a2c --model MiniGrid_size_A2C_image --env MiniGrid-Empty-8x8-v0 --frames 80000 --input image --load-optimizer-state True
python train.py --algo ppo --model MiniGrid_size_PPO_image --env MiniGrid-Empty-8x8-v0 --frames 80000 --input image  --load-optimizer-state True
python train.py --algo sr --model MiniGrid_size_SR_image --env MiniGrid-Empty-8x8-v0 --frames 80000 --input image --feature-learn curiosity --seed 1 --entropy-coef 0.01 --entropy-decay 0.9 --load-optimizer-state True

python train.py --algo a2c --model MiniGrid_size_A2C_ssp --env MiniGrid-Empty-8x8-v0 --frames 80000 --input ssp-xy  --seed 1 --load-optimizer-state True
python train.py --algo ppo --model MiniGrid_size_PPO_ssp --env MiniGrid-Empty-8x8-v0 --frames 80000 --input ssp-xy  --seed 1 --load-optimizer-state True
python train.py --algo sr --model MiniGrid_size_SR_ssp --env MiniGrid-Empty-8x8-v0 --frames 80000 --input ssp-xy --feature-learn none  --entropy-coef 0.01 --entropy-decay 0.9  --seed 1 --load-optimizer-state True

python train.py --algo a2c --model MiniGrid_size_A2C_image --env MiniGrid-Empty-16x16-v0 --frames 120000 --input image --load-optimizer-state True
python train.py --algo ppo --model MiniGrid_size_PPO_image --env MiniGrid-Empty-16x16-v0 --frames 120000 --input image  --load-optimizer-state True
python train.py --algo sr --model MiniGrid_size_SR_image --env MiniGrid-Empty-16x16-v0 --frames 120000 --input image --feature-learn curiosity --seed 1 --entropy-coef 0.01 --entropy-decay 0.9 --load-optimizer-state True

python train.py --algo a2c --model MiniGrid_size_A2C_ssp --env MiniGrid-Empty-16x16-v0 --frames 120000 --input ssp-xy  --seed 1 --load-optimizer-state True
python train.py --algo ppo --model MiniGrid_size_PPO_ssp --env MiniGrid-Empty-16x16-v0 --frames 120000 --input ssp-xy  --seed 1 --load-optimizer-state True
python train.py --algo sr --model MiniGrid_size_SR_ssp --env MiniGrid-Empty-16x16-v0 --frames 120000 --input ssp-xy --feature-learn none  --entropy-coef 0.01 --entropy-decay 0.9  --seed 1 --load-optimizer-state True

