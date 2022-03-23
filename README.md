# Team-Blackwell

This project aims to write a program that can balance a pole on a cart for 200 timesteps. This can be done using a simple Q-Learning Policy. We would like to improve the Q-Learning algorithm using methods from Bayesian Statistics.

Our first goal is to predict the future roll-out of states in the OpenAI Gym Cartpole environment. These prediction may improve the performance of an autonomous agent using a Q-Learning Policy. The predictions may be handled by the Gibbs sampling method. This allows the agent to determine the conditional probability of observing a specific future state given the current state and action. Hence, the choice of which action to take next is more informed than the simple Q-Learning Policy.

## Directory Information

#### /R

Analysis of states in Cartpole in order to gain insight about distribution parameters.

#### /Utils

Utility programs that simplify the process of accessing a Cartpole environment and plotting data. Includes:

-   `make_dataset.py` -- creates a `.csv` file of random actions with their respective observations for `n` episodes.

-   `analyze_random.py` -- allows user to observe how the simulation looks when random actions are taken. Can also be used to create a heatmap of what the ML Agent will see.

#### /LegacyMarkovAgent

Assorted programs from an old (poorly executed) experiment. Used for training an Agent using a sampling of the conditional state space in Cartpole via the Viterbi Algorithm. Also included are codes to evaluate performance and produce graphics.

-   evaluate_mm.py
-   markov_agent.py
-   markov_chain_recent.py
-   performance_markov_agent.png

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

To visualize the simulation while it is running (taking random actions) use the following code. (Note that if the position of the poll is too far then the simulation ends and resets.)

    python3 utils/analyze_random.py viz

The observation axes can also be examined for random episodes as a heatmap of total reward per timestep. Every frame of the simulation is a reward of 1, for a max of 200 frames.

    python3 utils/analyze_random.py plot

## Make a Dataset for Analysis

Runs the simulation for a number of episodes (default 50). Exports the observation data as a file (default `rand_state_acts.csv`)

    python3 utils/make_dataset.py
