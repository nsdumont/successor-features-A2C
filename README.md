![](https://github.com/nsdumont/successor-features-A2C/blob/main/drawing.png)

# Successor Features A2C
A pytorch implementation of successor features in an advantage actor-critic model, that works for both discrete and continuous action spaces. This code is adapted from https://github.com/lcswillems/rl-starter-files and https://github.com/lcswillems/torch-ac.
Included here is a novel learning rule for successor features, inspired by the off-line $\lambda$-return algorithm and generalized advantage estimation (GAE). 
Additionally a method for learning the state/feature representation, that can be interpreted as an intrinsic reward encouraging exploration, is included as well.

## Requirements 
See requirements file, or use the provided yml file to create a conda env:
<pre><code>conda env create -f environment.yml</code></pre>

## Examples
<pre><code>python train.py --algo sr --env MiniGrid-Empty-6x6-v0 --frames 50000 --input image --feature-learn curiosity</code></pre>
<pre><code>python train.py --algo sr --env MountainCarContinuous-v0 --frames 100000 --input flat --feature-learn curiosity</code></pre>

## Introduction

RL models can be divided into two categories, model-based and model-free, each of which has advantages and disadvantages. 
Model-based RL can be used for transfer learning and latent learning: an agent can learn a state transition model (without a reward signal) that can be used for navigation in novel environments as well as adapted to distal reward changes.
Consider a maze task with a goal at a certain location. With a model of the environment $Pr(s' | s,a)$, if the goal location changes, the learned model is still correct, and the algorithm only needs to update $r(s)$ at the affected states. 
However, action selection in model-based RL can be computational expensive. Value estimation must be done iteratively over the entire state space or a tree search method must be used. 
In contrast, model-free RL is slower to adapt to changes in the reward and transition functions. 
The simple local change in a goal location described above changes the value function and Q function globally. 
A model-free method therefore needs more experience to adapt, may forget its representations for the old goal location, and are incapable of latent learning. 


Successor representation (SR) provides a middle ground between model-based and model-free methods. 
In this paradigm one learns a reward function, $r(s)$, and a representation of long-term transitions in the environment, $M(s,s')$. 
To be exact, $M(s,s')$ is the expected number of discounted visits to state $s'$ given initial state $s$ and policy $\pi$:

$M^{\pi}(s,s') = E_{\pi}  [ \sum_{t=0}^{\infty} \gamma^t  1[s_t=s']  |  s_0=s  ] $

where $1[\cdot]$ is the indicator function. Just as the value function accumulates rewards discounted over an episode, the SR accumulates state occupancies. 
It represents temporal correlations between states. The successor representation follows a Bellman-like equation,
 
$M(s_t,s') = 1[s_t=s'] + \gamma M(s_{t+1},s')$

Thus, model-free algorithms for learning a value function (e.g. TD learning, eligibility traces, etc.) can be easily adapted for learning the SR. 
Furthermore, given the SR and a reward function we can easily compute state values,
 
$V^{\pi}(s) = \sum_{s'} M^{\pi}(s,s') r(s') $

$\mathbf{V}^{\pi} = M^{\pi}\mathbf{r} $
where $M^{\pi}$ is the matrix such that $M(s_i,s_j)$ is it's $(i,j)$ element, and $\mathbf{r}$ is the vector of state rewards. 
An alternative version of the SR for actions, $M(s,s',a)$ (the expected discounted future state occupancy starting from state $s$ and taking action $a$), can be used to compute the Q function instead. 
Thus, in the SR framework decisions can be made cheaply by selecting actions that maximize the Q function. 
This method can more quickly adapt to distal changes in the reward structure, facilitating transfer learning. 
However, changes in the state transition structure of the environment may be slower to propagate to the SR of all states. 

SR can be generalized for continuous states. Let $\Phi(s)$ can be a vector representation of a continuous state, called the state's features.
The features can be represented by a neural network and learned from raw state observations. 
The expected discounted sum of future features is called the successor feature function. 
$\mathbf{M}(s)$ will be the output of a neural network given $\Phi(s)$ as input. The TD error is

$\delta_t = \Phi(s_t)  + \gamma \mathbf{M}^{\pi}(s_{t+1})- \mathbf{M}^{\pi}(s_{t})$

If $\Phi(s)$ is a one-hot vector than the traditional SR can be recovered from this representation, $M(s,s’) = \mathbf{M}(s) \cdot \Phi(s’)$. 
Usually a linear function is used to represent the reward function when used alongside successor features.

$r(s) = \Phi(s) \cdot \mathbf{w}$

This is so that the value function can be recovered linearly: $V(s) = \mathbf{M}(s) \cdot \mathbf{w}$.

In past work using successor features, features were often learned using an autoencoder trained to reconstruct the raw state input, $\mathcal{L}_{\text{reconstruction}} = ||d(\Phi(s)) - s ||_2^2$ (where $\Phi$ is the output of the encoder and $d$ is the output of the decoder),
to ensure that the features sufficiently represent the state. This loss can be thought of as an intrinsic reward.

## Model
A diagram of the full advantage actor-critic generalized successor estimation (A2C-GSE) model is shown at the top of this readme. 
### Feature Learning
Learning a useful state representation while simultaneously learning values and/or policies is a major obstacle in deep RL. 
This is especially true in the context of transfer learning, where the features should be ideally be agnostic to particularities of the task, particularly in SR models where these features are needed to define the successor features.
The network that encodes the states cannot simply be optimized using the loss function of the SR as a solution where $\Phi(s) = 0$ and $\mathbf{M}^{\pi}(s)=0$, since all states would give zero error but would be useless for any task. 
Often the reconstruction error of an autoencoder is used to learn the feature representation. 
In this code two options are given for learning the feature representation. The first is reconstruction error and the second is curiosity, as defined in "Curiosity-driven Exploration by Self-supervised Prediction".
This model of curiosity is defined by the agent's ability to understand the affect of its actions on the environment.
In addition to an encoder, the model contains a forward model and inverse model.
The forward model takes the current feature and the action performed as input and predicts the next feature. 
The inverse model uses the current feature and the next feature to predict the action taken. 
For discrete actions, the probabilities of each action, $p(a_t | s_t, s_{t+1})$, is the output and a negative log-likelihood loss is used. 
For continuous actions, an estimated action $\hat{a}_t$ is the output and the L2 loss between $\hat{a}_t$ and $a_t$ is used as the inverse loss function. 
These extra loss terms can be though of as intrinsic rewards which ensure that the features contain information about the aspects of the state that the agent can affect with its actions.
In addition, the use of a feature norm loss is included. 

### Generalized Successor Estimation \& Advantage Functions
Generalized advantage estimation (GAE) is used to compute the policy loss. 
An exponentially weighted average of the $n$-step advantage estimates across the episode is called the generalized advantage estimator:

$A_t = (1-\lambda) \sum_{l=0} \lambda^l A^{(l+1)}_t$

$= \sum_{l=0} (\gamma \lambda)^l (r_{t+l} + \gamma V^{\pi}(s_{t+1+l})- V^{\pi}(s_{t+l}))$

where $\lambda$ is a hyperparameter. When $\lambda=0$ this reduces to the standard TD advantage and when $\lambda=1$ it is the Monte Carlo estimate of the advantage.

This equation can be rewritten using the successor features:

$A_t = \sum_{l=0}^{T-1} (\gamma \lambda)^l (\Phi(s_{t+l}) \cdot  \mathbf{w} + \gamma \mathbf{M}^{\pi}(s_{s+1+l}) \cdot \mathbf{w}- \mathbf{M}^{\pi}(s_{t+l}) \cdot  \mathbf{w})$
$= \left [ \sum_{l=0}^{T-1} (\gamma \lambda)^l \boldsymbol{\delta}_{t+l}^{SR} \right ] \cdot  \mathbf{w} $
$=  \mathbf{M}^{GSE(\gamma, \lambda)}_t \cdot  \mathbf{w}$

I call the quantity in brackets the generalized successor estimation (GSE). It can be computed for an episode and used to learn the SR itself. 
This is analogous to the offline TD($\lambda$) learning rule for value functions except the estimator is a vector quantity. 
Thus, the GSE can be computed to update the SR and the GSE can be fed into the linear reward network to approximate the GAE. This can then be used to compute the policy loss.

