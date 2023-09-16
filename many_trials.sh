
#python train.py --algo sr --env MiniGrid-Empty-6x6-v0 --frames 50000 -a2c --lr_sr 0.01

#runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo sr --env MiniGrid-Empty-6x6-v0 --frames 50000 --input ssp-minigrid-xy --feature-learn none --lr_sr 0.01 --ssp-h 1', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)
#runfile('/home/ns2dumon/Documents/Github/successor-features-A2C/train.py', args='--algo sr --env MiniWorld-TMazeLeft-v0 --frames 50000 --input ssp-miniworld-xy --feature-learn none --lr_sr 0.01 --ssp-h 1 --procs 1', wdir='/home/ns2dumon/Documents/Github/successor-features-A2C', post_mortem=True)
# Parse arguments

for alg in "a2c" "ppo" "sr"
do
    for input_type in "image" "ssp-minigrid-xy"
    do
        if [ $input_type="image" ]
        then
           flearn="curiosity"
        else
           flearn="none"
        fi

    	for seed in 0 1 2 3 4 5 6 7 8 9
    	do
        	dest_dir="MiniGrid-Empty-6x6-"$alg"-"$input_type"-"$seed
    		python train.py --env MiniGrid-Empty-8x8-v0 --algo $alg --model $dest_dir --input $input_type --feature-learn $flearn --lr_sr 0.01 --batch-size 500 --lr 0.001
    	done
    done
done

