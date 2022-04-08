from cmath import inf
import imp
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('bmh')

gym.envs.register(
    id='CartPoleExtraLong-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=250,
    reward_threshold=-110.0,
)
env = gym.make('CartPoleExtraLong-v0')
#env = gym.make('CartPole-v0')
observe_training = False
TARGET_REWARD = 200
EPSILON_MIN = 0.1
NUM_BINS = 10 # 10
ALPHA = 0.01 # 0.01
GAMMA = 0.9
'''
    TODO: Fix the Q matrix s.t. it creates the necissary bins
          Record the number of times each state is updated
'''

EPOCHS = 2000

obs_space = 4
action_space = env.action_space.n

class DQN:
    def __init__(self, num_bins, observation):

        self.obs_space = obs_space
        
        self.bins = self.get_bins(num_bins)
        #self.init_Q_matrix(observation)
        self.Q = {}
        self.digitize(observation)

    def get_bins(self, num_bins):
        # Make 10 x state_depth matrix,  each column elem is range/10
        # Digitize using bins ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # obs[0]--> min: -0.11 | max: 0.51 | median : 0.00
        # obs[1]--> min: -1.41 | max: 1.88 | median : 0.16
        # obs[2]--> min: -0.21 | max: 0.21 | median :-0.04
        # obs[3]--> min: -2.66 | max: 2.29 | median :-0.19

        bins = []
        ranges = [4.8, 5, 0.418, 5]
        #ranges = [[-0.11,0.51], [-1.41,1.88],[-0.21, .21], [-2.66,2.29]]
        
        for i in range(self.obs_space):
            start, stop = -ranges[i], ranges[i]

            # Using Max and Min values observed in random episodes
            #start, stop = ranges[i][0], ranges[i][1]
            buckets = np.linspace(start, stop, num_bins)
            bins.append(buckets)
    
        return bins
    
    def add_Q_state(self, obs_string):
        if obs_string not in self.Q:
            #self.Q[obs_string] = np.random.normal(0,0.001,(2))
            self.Q[obs_string] = np.zeros((2))

    def init_Q_matrix(self, obs):
        ''' DEPRECIATED '''
        assert(len(obs)==self.obs_space)
        states = []
        for i in range(10**(self.obs_space)):
            #populates state with left padded numbers as str
            states.append(str(i).zfill(self.obs_space))

        self.Q = {}
        for state in states:
            self.Q[state] = np.zeros((2))

    def digitize(self, arr):
        # distribute each elem in state to the index of the closest bin
        state = np.zeros(len(self.bins))
        for i in range(len(self.bins)):
            state[i] = np.digitize(arr[i], self.bins[i])
        state = ''.join(str(int(elem)) for elem in state)
        self.add_Q_state(state)
        return state
    
    def get_action(self, state):
        return np.argmax(self.Q[self.digitize(state)])

    def evaluate_utility(self, state):
        return np.max(self.Q[self.digitize(state)])
    
    def update_policy(self, state, state_next, action, reward):
        state_value = self.evaluate_utility(state)
        
        action = self.get_action(state)
        reward_next = self.evaluate_utility(state_next)

        state_value += ALPHA*(reward + GAMMA * reward_next - state_value)

        state = self.digitize(state)
        self.Q[state][action] = state_value


    def get_state_stats(self):
        for key, value in enumerate(len(self.Q)):
            print("\nElem:",key,end=" ")            
            print("Range: [%s, %s]" % (min(value), max(value)),
                  "STDDEV:", round(np.std(value), 3), "Count:" , len(value))
    
def play_episode(agent, epsilon=0.2, viz=False):
    total_reward = 0
    timestep = 0
    all_states = [env.reset(seed=451)]
    terminal = False
    all_actions = []
    action = 0


    max_rwd = -200
    while not terminal:
        if viz: env.render()
        
        if np.random.random() < epsilon:
            act = env.action_space.sample()
        else:
            action = agent.get_action(all_states[-1])
        
        state_next, reward, terminal, info = env.step(action)

        total_reward += reward

        if terminal and timestep < TARGET_REWARD:
            reward = -300
        
        if reward > 300:
            import pdb; pdb.set_trace()
        
        agent.update_policy(all_states[-1], state_next, action, reward)
        all_states.append(state_next)
        all_actions.append(action)
        timestep += 1

    all_actions = np.array(all_actions)
    # Ignore last obs since next_obs will be null
    return all_states[:-1], all_actions, total_reward, timestep

def train(epochs=2000, agent=False):
    if not agent: agent = DQN(NUM_BINS, env.reset())
    total_duration = []
    total_reward = []
    digi_states = []
    all_actions = []

    for ep in range(epochs):
        epsilon = max(EPSILON_MIN, np.tanh(-ep/(epochs/2))+1)

        ep_states, ep_acts, reward, timesteps = play_episode(agent, epsilon)
        total_duration.append(timesteps)
        total_reward.append(reward)
        
        mu_rew = 0
        if (ep % 200) == 0:
            if ep >= 200:
                mu_rew=np.mean(total_reward[-200])
                mu_rew=round(mu_rew, 3)
            else:
                mu_rew = total_reward[-1]
            epsi=round(epsilon, 4)
            perc_done=(ep*100)//epochs
            print(f"Ep:{ep} | {epochs}, %: {perc_done}" + \
                  f", Epsi: {epsi}, Avg Rwd: {mu_rew}")

        # Calculate the state values
        for idx, state in enumerate(ep_states):
            ep_states[idx] = agent.digitize(state)
        digi_states.append(ep_states)
        all_actions.append(ep_acts)

        # End training early if max performance
        if mu_rew >= TARGET_REWARD:
            break
    
    return digi_states, all_actions, total_reward, total_duration, agent

def observe(agent, N=15):
    [play_episode(agent, epsilon=-1, viz=True) for ep in range(N)]

def plot_running_avg(reward_arr):
    # TODO: calculate and graph via pandas

    plt.title("Average Reward with Q-Learning By Episode (CartPole)")
    N = len(reward_arr)
    #init unitialized array
    # (faster than np.zeros)
    running_avg = np.empty(N)

    for t in range(100, N):
        running_avg[t] = np.mean(reward_arr[t-100: t+1])
        if running_avg[t] > 300:
            import pdb; pdb.set_trace()

    plt.plot(running_avg, color="purple", label="Q-Learning Running Average")
    plt.xlabel('Training Time (episodes)', fontsize=18)
    plt.ylabel('Average Reward per Episode', fontsize=16)
    plt.legend()
    plt.show()

def save_stats(info, filename):
    
    df_array = []
    for episode, obs in enumerate(info['state']):
        N = len(obs)
        ones = np.ones(N)
        ep_dict = {"state":obs}
        ep_dict['timestep'] = np.arange(N)
        ep_dict['episode'] = ones * episode
        ep_dict['action'] = info['action'][episode]
        ep_dict['reward'] = ones * info['ep_reward'][episode]
        ep_dict['duration'] = ones * info['ep_duration'][episode]
        df_array.append(pd.DataFrame(ep_dict))
    
    df = pd.concat(df_array)
    df.to_csv(f"{filename}_stats.csv", index=False)

    Q_table = {"state":[], "act0":[], "act1":[], "argmax":[]}
    for index in sorted(list(info['Agent'].Q.keys())):
        state_values = info['Agent'].Q[index]
        Q_table['argmax'].append(np.argmax(state_values))
        Q_table['act0'].append(state_values[0])
        Q_table['act1'].append(state_values[1])
        Q_table['state'].append(index)
    
    Q_table = pd.DataFrame(Q_table)
    Q_table.to_csv(f"{filename}_Qtable.csv", index=True)




if __name__ == "__main__":
    #EPOCHS = 10
    info = train(epochs = EPOCHS)

    labels = ["state", "action", "ep_reward", "ep_duration", "Agent"]
    info = dict(zip(labels, info))
    plot_running_avg(info['ep_reward'])
    save_stats(info, "StringQ_rngBins")
    
    observe(info["Agent"], N=5)
    #plt.plot(random_rwds, color="gray", label="Random Moves Running Average")

    