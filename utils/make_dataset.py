import gym
import numpy as np
import pandas as pd

env = gym.make('CartPole-v1')

def play_random(viz=True):
    obs = [env.reset(seed=451)]
    next_obs = []
    actions = []
    total_reward = 0
    terminal = False

    while not terminal:
        if viz: env.render()
        act = env.action_space.sample()
        observation, reward, terminal, info = env.step(act)
        actions.append(act)
        obs.append(observation)
        next_obs.append(observation)

        total_reward += reward

    # Ignore last obs since next_obs will be null        
    return obs[:-1], next_obs, actions, total_reward

def episodes(epochs, viz=False):

    frames, next_frames, actions, rewards = [],[], [], []
    for ep in range(epochs):
        X = play_random(viz)
        frames.append(X[0])
        next_frames.append(X[1])
        actions.append(X[2])
        rewards.append(X[3])
        
    return {'obs':frames, 'act':actions, 'next_obs':next_frames, 'reward':rewards}

def save_observations(filename, epochs=100):
    ''' Exec num epochs episodes taking random actions 
        Save info about simulation as .csv
    '''
    frames_rewards = episodes(epochs)
    df = pd.DataFrame(frames_rewards)
    actions = df['act'].values
    rewards = df['reward'].values
    observations = df['obs'].values
    next_observation = df['next_obs'].values

    df_arr  = []
    
    for i, ep_obs in enumerate(observations):
        ep_obs = np.array(ep_obs)
        ep_obs_next = np.array(next_observation[i])

        ep_len = len(ep_obs[:, 0])
        
        ep_dict = {'timestep':np.arange(ep_len)}
        ep_dict['episode'] = np.ones(ep_len) * i
        # Assumes reward for each timestep is total reward
        ep_dict['reward'] = np.ones(ep_len) * rewards[i]
        ep_dict['act'] = actions[i]
        
        for ax in range(len(ep_obs[0])):
            ep_dict[f'obs_ax{ax}'] = ep_obs[:, ax]
            ep_dict[f'next_ax{ax}'] = ep_obs_next[:, ax]

        ep_dict = pd.DataFrame(ep_dict)
        df_arr.append(ep_dict)
    
    

    df = pd.concat(df_arr)
    df.to_csv(filename, index=False)

'''
observation                 Min         Max
  0	Cart Position             -4.8            4.8
  1	Cart Velocity             -Inf            Inf
  2	Pole Angle                 -24           24
  3	Pole Velocity At Tip      -Inf            Inf
'''

if __name__ == "__main__":
    save_observations('rand_state_acts.csv', epochs=50)