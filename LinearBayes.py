from re import T
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')

class LinearBayesAgent:
    '''
    Factor Analysis Loadings
        ax0 -0.696
        ax1 -0.964 
        ax2  0.742 
        ax3  0.998 

    Loadings SD
        1.380598
    '''
    def __init__(self):
        # Projection parameters for state subspace from Factor Analysis
        self.loadings = np.array([-0.696,-0.964,0.742,0.998]).T

        rng, num_bins = 1.38, 100
        # Discrete representation of states
        self.bucket = self.init_bucket(rng, num_bins) 

        self.Q = self.init_Q() # Matrix of state-value pairs by action
        self.lr = 0.1 # Learning Rate
        self.gamma = 0.9 # Discount factor for future reward
        self.epsilon = 0.7 # Epsilon Greedy parameter (annealing)
        

    def play_episode(self, train=True, viz=False):
        obs = [env.reset(seed=451)]
        terminal = False
        total_reward = 0
        actions = []
        while not terminal:
            if viz: env.render()

            act = self.get_action(obs[-1], train)

            new_state, reward, terminal, info = env.step(act)
            self.update_policy(obs[-1], new_state, act, reward)
            
            actions.append(act)
            obs.append(new_state)
            total_reward += reward

            if terminal:
                # Attenuate random action prob
                #self.epsilon *= 0.9
                # Set Q-Value for terminal state as total reward
                #_, Q_idx = self.get_Q(obs[-1], act)
                #self.Q[Q_idx][act] = total_reward
                pass

        obs = np.dot(obs, self.loadings)
        return obs, total_reward, actions
    
    def init_bucket(self, rng, num_bins):
        start, stop = -rng, rng
        bucket = np.linspace(start, stop, num_bins)        
        return bucket
    
    def init_Q(self):
        n = len(self.bucket)
        return dict(zip(np.arange(0, n), np.ones((n, 2))))

    def get_Q(self, state, action):
        # Project states vector onto loadings subspace
        factor = np.dot(state, self.loadings)
        # Calculate position of state in Q vector
        for idx, elem in enumerate(self.bucket):
            if factor < elem or idx == len(self.bucket) - 1:
                # Return Q-value (for state and action) and Q index
                return self.Q[idx][action], idx

    def pred_next(self):
        # TODO: Use BLM
        pass

    def get_action(self, state, training=True):
        rand_prob = np.random.random()
        if training and rand_prob < self.epsilon:
            # Select an action at random
            return env.action_space.sample()
        else:
            # Return the action with maximum Q-value
            _, q_idx = self.get_Q(state, 0)
            return np.argmax(q_idx)
    
    def update_policy(self, state, state_next, action, reward):
        state_value, Q_idx = self.get_Q(state, action)
        # TODO: predict state_next from BLM
        value_next, Q_idx = self.get_Q(state_next, action)
        value_next = np.max(self.Q[Q_idx])
        # Update state_value using bellman equation
        state_value += self.lr*(reward + (self.gamma * value_next) - state_value)

def run_sim(agent, episodes=100, train=True, plot=False):
    data = {"obs":[], "reward":[],"actions":[]}
    for ep in range(episodes):
        obs, reward, actions = agent.play_episode(train=train)
        data["actions"].append(actions)
        data["reward"].append(reward)
        data["obs"].append(obs)
        if ep % episodes//10 == 0:
            print(reward)
    
    if plot:
        df = pd.DataFrame(data)
        df.plot()
        plt.title(f"Training={train}")
        plt.show()

        
    

if __name__ == "__main__":
    agent = LinearBayesAgent()
    #import pdb; pdb.set_trace()
    run_sim(agent, episodes=10000, plot=True)
    df = pd.DataFrame(agent.Q)
    #import pdb; pdb.set_trace()
    df.to_csv("Q_table.csv", index=False)
    #plt.scatter(np.arange(100), df.values[0])
    #plt.scatter(np.arange(100), df.values[1])
    #plt.show()
    run_sim(agent, episodes=100, train=False)
    

