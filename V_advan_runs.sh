


python3 train.py --algo sr --model MiniGrid_Goal_SR_image2 --env MiniGrid-EmptyGoal-6x6-v0 --frames 80000 --input image --feature-learn reconstruction --target-update 10 --recon-loss-coef 5 --entropy-coef 0.005 --batch-size 100 --frames-per-proc 100 --seed 1 --env-args "{switch_prob: 0}"

python3 train.py --algo sr --model MiniGrid_Goal_SR_ssp2 --env MiniGrid-EmptyGoal-6x6-v0 --frames 60000 --input ssp --feature-learn curiosity --target-update 1 --recon-loss-coef 0 --entropy-coef 0.005 --batch-size 300 --frames-per-proc 10 --seed 1 --env-args "{switch_prob: 0}"




python3 train.py --algo sr --model MiniGrid_Goal_SR_image2 --env MiniGrid-EmptyGoal-6x6-v0 --frames 160000 --input image --feature-learn reconstruction --target-update 10 --recon-loss-coef 5 --entropy-coef 0.005 --batch-size 100 --frames-per-proc 100 --load-optimizer-state 0 --seed 2 --env-args "{switch_prob: 0}" 

python3 train.py --algo sr --model MiniGrid_Goal_SR_ssp2 --env MiniGrid-EmptyGoal-6x6-v0 --frames 160000 --input ssp --feature-learn curiosity --target-update 1 --recon-loss-coef 0 --entropy-coef 0.005 --batch-size 300 --frames-per-proc 10 --load-optimizer-state 0 --seed 2 --env-args "{switch_prob: 0}"




python3 train.py --algo sr --model MiniGrid_Goal_SR_image2 --env MiniGrid-EmptyGoal-6x6-v0 --frames 240000 --input image --feature-learn reconstruction --target-update 10 --recon-loss-coef 5 --entropy-coef 0.005 --batch-size 100 --frames-per-proc 100 --load-optimizer-state 0 --seed 3 --env-args "{switch_prob: 0}"

python3 train.py --algo sr --model MiniGrid_Goal_SR_ssp2 --env MiniGrid-EmptyGoal-6x6-v0 --frames 240000 --input ssp --feature-learn curiosity --target-update 1 --recon-loss-coef 0 --entropy-coef 0.005 --batch-size 300 --frames-per-proc 10 --load-optimizer-state 0 --seed 3 --env-args "{switch_prob: 0}"



python3 train.py --algo sr --model MiniGrid_size_SR_image2 --env MiniGrid-Empty-6x6-v0 --frames 80000 --input image --feature-learn reconstruction --target-update 10 --recon-loss-coef 5 --entropy-coef 0.005 --batch-size 300 --frames-per-proc 100 --seed 1
python3 train.py --algo sr --model MiniGrid_size_SR_ssp2 --env MiniGrid-Empty-6x6-v0 --frames 80000 --input ssp --feature-learn curiosity --target-update 1 --recon-loss-coef 0 --entropy-coef 0.005 --batch-size 300 --frames-per-proc 10 --seed 1 


python3 train.py --algo sr --model MiniGrid_size_SR_image2 --env MiniGrid-Empty-8x8-v0 --frames 160000 --input image --feature-learn reconstruction --target-update 10 --recon-loss-coef 5 --entropy-coef 0.005 --batch-size 300 --frames-per-proc 100 --load-optimizer-state 0 --seed 1
python3 train.py --algo sr --model MiniGrid_size_SR_ssp2 --env MiniGrid-Empty-8x8-v0 --frames 160000 --input ssp --feature-learn curiosity --target-update 1 --recon-loss-coef 0 --entropy-coef 0.005 --batch-size 300 --frames-per-proc 10 --load-optimizer-state 0 --seed 1 




python3 train.py --algo sr --model MiniGrid_size_SR_image2 --env MiniGrid-Empty-16x16-v0 --frames 240000 --input image --feature-learn reconstruction --target-update 10 --recon-loss-coef 5 --entropy-coef 0.005 --batch-size 300 --frames-per-proc 100 --load-optimizer-state 0 --seed 1
python3 train.py --algo sr --model MiniGrid_size_SR_ssp2 --env MiniGrid-Empty-16x16-v0 --frames 240000 --input ssp --feature-learn curiosity --target-update 1 --recon-loss-coef 0 --entropy-coef 0.005 --batch-size 300 --frames-per-proc 10 --load-optimizer-state 0 --seed 1



python3 train.py --algo sr --model MiniGrid_NoGoal_SR_image2 --env MiniGrid-EmptyNoGoal-6x6-v0 --frames 50000 --input image --feature-learn reconstruction --target-update 10 --recon-loss-coef 5 --entropy-coef 0.1 --batch-size 100 --frames-per-proc 100  --seed 1

python3 train.py --algo sr --model MiniGrid_NoGoal_SR_ssp2 --env MiniGrid-EmptyNoGoal-6x6-v0 --frames 50000 --input ssp --feature-learn curiosity --target-update 1 --recon-loss-coef 0 --entropy-coef 0.1 --batch-size 300 --frames-per-proc 10  --seed 1




python3 train.py --algo sr --model MiniGrid_NoGoal_SR_image2 --env MiniGrid-Empty-6x6-v0 --frames 130000 --input image --feature-learn reconstruction --target-update 10 --recon-loss-coef 5 --entropy-coef 0.005 --batch-size 100 --frames-per-proc 100 --load-optimizer-state 0  --seed 1

python3 train.py --algo sr --model MiniGrid_NoGoal_SR_ssp2 --env MiniGrid-Empty-6x6-v0 --frames 130000 --input ssp --feature-learn curiosity --target-update 1 --recon-loss-coef 0 --entropy-coef 0.005 --batch-size 300 --frames-per-proc 10 --load-optimizer-state 0  --seed 1



