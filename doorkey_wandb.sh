
for env in "MiniGrid-DoorKey-6x6-v0"
do
    for (( seed = 0 ; seed < $1 ; ++seed ))
    do
        echo "Seed: $seed"
        python train.py --env $env --algo a2c --frames 100000 --wrapper ssp-view --input flat  --seed $seed --wandb --wandb-project-name ssp-rl --wandb-entity nicole-s-dumont --wandb-tags $env ssp-obs3
    done
done

# python train.py --env $env --algo a2c --frames 100000 --wrapper xy --input flat --procs 1 --frames-per-proc 100 --seed $seed --wandb --wandb-project-name ssp-rl --wandb-entity nicole-s-dumont --wandb-tags $env xy-obs
