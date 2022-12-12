This repository contains code to train noisy quantum agents for variational Q-learning and policy gradients, as used in arXiv:xxxx.xxxx .
The two environments implemented here are [CartPole from the OpenAI Gym](https://github.com/openai/gym/wiki/CartPole-v0) and the Traveling Salesperson Environment from [arXiv:2205.06109](https://arxiv.org/pdf/2205.06109). 

To run a noisy quantum agent, pick one of the `run_`-files and specify the desired type of noise with the appropriate hyperparameters.

## Shot noise

| HP                     | meaning                                                             |
|------------------------|---------------------------------------------------------------------|
| n_shots                | number of measurements to estimate one expectation value            |
| ucb_alg                | use UCB-inspired algorithm for flexible shot allocation             |
| ucb_alg_init_shots     | number of measurements of first iteration of UCB-inspired algorithm |
 | ucb_alg_shot_increment | number of measurements in each consecutive step of algorithm        |
| ucb_alg_max_shots      | maximum number of allowed shots                                     |


## Coherent noise

| HP  | meaning                     |
|-----|-----------------------------|
| param_perturbation    | σ for Gaussian perturbation |


## Incoherent noise

| HP                    | meaning                                                    |
|-----------------------|------------------------------------------------------------|
| n_trajectories        | number of trajectories for Monte Carlo simulation of noise |
| single_qb_depol_error | single-qubit depolarization noise strength                 |
| two_qb_depol_error    | two-qubit depolarization noise strength                    |
| bitflip_error         | bitflip error strength                                     |
| amplitude_damp_error  | amplitude damping noise strength                           |
