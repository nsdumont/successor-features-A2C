import argparse
import yaml
import numpy as np
import optuna
from train import run

from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_rank
from optuna.visualization import plot_slice
from optuna.visualization import plot_timeline

                

def optim(env,algo,wrapper,input,frames, n_seeds,n_trials, domain_dim=None, discount=0.99, env_args={}, initial_params=None):
    def objective(trial):
        # discount = trial.suggest_categorical("discount", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
        max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5, 10])
        gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
        lr = trial.suggest_float("lr", 1e-5, 1, log=True)
        entropy_coef = trial.suggest_float("entropy_coef", 0.00000001, 0.1, log=True)
        entropy_decay = trial.suggest_float("entropy_decay", 0.00000001, 1.0, log=True)
        procs = trial.suggest_categorical("procs", [1,2,4,6])
        value_loss_coef = trial.suggest_float("value_loss_coef", 0, 1)

        actor_hidden_size = trial.suggest_categorical('actor_hidden_size', [32, 64, 128, 256])
        critic_hidden_size= trial.suggest_categorical('critic_hidden_size', [32, 64, 128, 256])
        feature_hidden_size= trial.suggest_categorical('feature_hidden_size', [32, 64, 128, 256])
        
        if algo=='ppo':
            clip_eps=trial.suggest_float('clip_eps',0.01,0.8)
            batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
            epochs = trial.suggest_categorical("epochs", [1, 5, 10, 20])
        else:
            clip_eps=0.2
            batch_size=256
            epochs=4
        
        if 'ssp' in wrapper:
            dissim_coef = 0.0#trial.suggest_float("dissim_coef", 0.0, 0.1)
            ssp_dim = trial.suggest_categorical("ssp_dim", [2*(domain_dim+1)*(i**2) + 1 for i in range(2,8)])
            ssp_h = trial.suggest_float("ssp_h", 0.0001, 100, log=True)
        else:
            dissim_coef=0.0
            ssp_dim=1
            ssp_h=1.0
            
        final_returns = np.zeros(n_seeds)
        for i in range(n_seeds):
            seed =  np.random.randint(0, 10000)
            _, final_return = run(env=env, seed=seed, algo=algo, wrapper=wrapper,
                                  frames=frames, env_args = env_args, input=input,
                                  verbose=False, save_interval=-1, plot=False,
                                  discount=discount, max_grad_norm=max_grad_norm, gae_lambda=gae_lambda,
                                  lr=lr, entropy_coef=entropy_coef, entropy_decay=entropy_decay, procs=procs,
                                  value_loss_coef=value_loss_coef, actor_hidden_size=actor_hidden_size, 
                                  critic_hidden_size=critic_hidden_size, feature_hidden_size=feature_hidden_size,
                                  clip_eps=clip_eps, batch_size=batch_size, epochs=epochs,
                                  dissim_coef=dissim_coef,ssp_dim=ssp_dim, ssp_h=ssp_h);
            final_returns[i] = final_return

        return np.mean(final_returns)
    
                    
    
    study = optuna.create_study(direction="maximize")
    if initial_params is not None:
        study.enqueue_trial(initial_params)
        
    study.optimize(objective, n_trials=n_trials)
    
    plot_optimization_history(study)
    plot_param_importances(study)
    plot_slice(study)
    return study.best_params

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", required=True,
                        help="algorithm to use: a2c | ppo | sr | sr-ppo (REQUIRED)")
    parser.add_argument("--env", required=True,
                        help="name of the environment to train on (REQUIRED)")
    parser.add_argument("--input", type=str, default="auto",
                        help="format of input:  auto | image | flat | ssp | none (default: auto)")
    parser.add_argument("--wrapper", type=str, default="none",
                        help="format of input:  none | ssp-xy | ssp-auto | one-hot | FullyObsWrapper | RGBImgObsWrapper | OneHotPartialObsWrapper | DirectionObsWrapper (default: non)")
    parser.add_argument("--env-args", type=yaml.load, default={'render_mode': 'rgb_array'},
                        help="")
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="num seeds")
    parser.add_argument("--frames", type=int, default=10**7,
                        help="number of frames of training (default: 1e7)")
    parser.add_argument("--domin-dim", type=int, default=1,
                        help="")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="")
    args = parser.parse_args()
    
    best_params = optim(args.env,args.algo,args.wrapper,args.input,args.frames, args.n_seeds, args.n_trials, args.domain_dim, args.env_args)
    print(best_params)
    