# Team-Blackwell

This project aims to write a program that can balance a pole on a cart for 200 timesteps. This can be done using a simple Q-Learning Policy. We would like to improve the Q-Learning algorithm using methods from Bayesian Statistics.

Our first goal is to predict the future roll-out of states in the OpenAI Gym Cartpole environment. These prediction may improve the performance of an autonomous agent using a Q-Learning Policy. The predictions may be handled by the Gibbs sampling method. This allows the agent to determine the conditional probability of observing a specific future state given the current state and action. Hence, the choice of which action to take next is more informed than the simple Q-Learning Policy.

## Directory Information

### /R

Analysis of states in Cartpole in order to gain insight about distribution parameters.

### /Utils

Utility programs that simplify the process of accessing a Cartpole environment and plotting data. Includes `make_dataset.py` that creates a `.csv` file of random actions with their respective observations for `n` episodes.

### /LegacyMarkovAgent

Assorted programs from an old (poorly executed) experiment. Used for training an Agent using a sampling of the conditional state space in Cartpole via the Viterbi Algorithm. Also included are codes to evaluate performance and produce graphics. - evaluate_mm.py - markov_agent.py - markov_chain_recent.py - performance_markov_agent.png

## Installing Gym

Use the following to install requirements in a virtual environment at the terminal (python3.6 or greater). 

``` bash
virtualenv venv && source venv/bin/activate && pip install -r requirements.txt
```

Subsequent BASH sessions may need the virtual environment to be activated with the following command.

``` bash
source venv/bin/activate
```

## Executing the Simulation
