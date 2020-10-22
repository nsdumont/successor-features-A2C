python3 train.py --algo a2c --model MiniGrid_Goal_A2C --env MiniGrid-EmptyGoal-6x6-v0 --frames 80000 --seed 1 --env-args "{switch_prob: 0}"
python3 train.py --algo ppo --model MiniGrid_Goal_PPO --env MiniGrid-EmptyGoal-6x6-v0 --frames 80000 --seed 1 --env-args "{switch_prob: 0}"

python3 train.py --algo sr --model MiniGrid_Goal_SR_image --env MiniGrid-EmptyGoal-6x6-v0 --frames 80000 --input image --feature-learn curiosity --target-update 10 --recon-loss-coef 5 --entropy-coef 0.005 --batch-size 100 --frames-per-proc 100 --seed 1 --env-args "{switch_prob: 0}"

python3 train.py --algo sr --model MiniGrid_Goal_SR_ssp --env MiniGrid-EmptyGoal-6x6-v0 --frames 60000 --input ssp --feature-learn curiosity --target-update 1 --recon-loss-coef 5 --entropy-coef 0.005 --batch-size 300 --frames-per-proc 10 --seed 1 --env-args "{switch_prob: 0}"



python3 train.py --algo a2c --model MiniGrid_Goal_A2C --env MiniGrid-EmptyGoal-6x6-v0 --frames 160000 --seed 2 --env-args "{switch_prob: 0}"
python3 train.py --algo ppo --model MiniGrid_Goal_PPO --env MiniGrid-EmptyGoal-6x6-v0 --frames 160000 --seed 2 --env-args "{switch_prob: 0}"
python3 train.py --algo sr --model MiniGrid_Goal_SR_image --env MiniGrid-EmptyGoal-6x6-v0 --frames 160000 --input image --feature-learn curiosity --target-update 10 --recon-loss-coef 5 --entropy-coef 0.005 --batch-size 100 --frames-per-proc 100 --load-optimizer-state 0 --seed 2 --env-args "{switch_prob: 0}" 

python3 train.py --algo sr --model MiniGrid_Goal_SR_ssp --env MiniGrid-EmptyGoal-6x6-v0 --frames 160000 --input ssp --feature-learn curiosity --target-update 1 --recon-loss-coef 5 --entropy-coef 0.005 --batch-size 300 --frames-per-proc 10 --load-optimizer-state 0 --seed 2 --env-args "{switch_prob: 0}"



python3 train.py --algo a2c --model MiniGrid_Goal_A2C --env MiniGrid-EmptyGoal-6x6-v0 --frames 240000 --seed 3 --env-args "{switch_prob: 0}"
python3 train.py --algo ppo --model MiniGrid_Goal_PPO --env MiniGrid-EmptyGoal-6x6-v0 --frames 240000 --seed 3 --env-args "{switch_prob: 0}"
python3 train.py --algo sr --model MiniGrid_Goal_SR_image --env MiniGrid-EmptyGoal-6x6-v0 --frames 240000 --input image --feature-learn curiosity --target-update 10 --recon-loss-coef 5 --entropy-coef 0.005 --batch-size 100 --frames-per-proc 100 --load-optimizer-state 0 --seed 3 --env-args "{switch_prob: 0}"

python3 train.py --algo sr --model MiniGrid_Goal_SR_ssp --env MiniGrid-EmptyGoal-6x6-v0 --frames 240000 --input ssp --feature-learn curiosity --target-update 1 --recon-loss-coef 5 --entropy-coef 0.005 --batch-size 300 --frames-per-proc 10 --load-optimizer-state 0 --seed 3 --env-args "{switch_prob: 0}"





