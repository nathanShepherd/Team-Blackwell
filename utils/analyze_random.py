import gym
import numpy as np
import pandas as pd
import seaborn as sb
from sys import argv
import matplotlib.pyplot as plt


env = gym.make('CartPole-v1')
#plt.style.use('bmh')
sb.set_style('darkgrid')


def play_random(viz=True):
    obs = [env.reset()]
    total_reward = 0
    terminal = False

    while not terminal:
        if viz: env.render()
        action = env.action_space.sample()
        observation, reward, terminal, info = env.step(action)
        obs.append(observation)
        total_reward += reward
        
    return obs, total_reward

def episodes(epochs, viz=False):

    frames, rewards = [],[]
    for ep in range(epochs):
        X = play_random(viz)
        frames.append(X[0])
        rewards.append(X[1])
        
    return {'obs':frames, 'reward':rewards}

def stack_figures(X, Y, shape=(2, 1), ylabel=['y1', 'y2'], xlabel=['x']):
    if shape[0] * shape[1] != len(Y):
        raise NameError('Length of Y must be num of subplts')
    fig, ax = plt.subplots(shape[0])
    
    plt_num = 0
    for row in range(1, shape[0] + 1):
        for axes in ax:
            axes.plot(X[plt_num], Y[plt_num])
            plt_num += 1


    

def heatmap(tensor, title=''):
    print(tensor, tensor.shape)
    
    fig, ax = plt.subplots(ncols=tensor.shape[0],
                                         nrows=1, sharey=True)
    
    
    for i in range(len(tensor)):
        obs_matrix = np.array(tensor[i])
        print(obs_matrix.shape)
        ax[i].imshow(obs_matrix, cmap='hot',
                             interpolation='nearest')
        plt.tight_layout()
        '''
        omat = pd.DataFrame(obs_matrix)
        print(omat.head())
        omat.plot()
        '''
    
    plt.suptitle(title, verticalalignment='center')
    
    plt.show()
    
def save_observations(filename, epochs=1000):
    
    frames_rewards = episodes(epochs)
    df = pd.DataFrame(frames_rewards)
    rewards = df['reward'].values
    observations = df['obs'].values

    df_arr  = []
    
    for i, ep_obs in enumerate(observations):
        ep_obs = np.array(ep_obs)
        ep_len = len(ep_obs[:, 0])

        
        ep_dict = {'timestep':np.arange(ep_len)}
        ep_dict['episode'] = np.ones(ep_len) * i
        # Assumes reward for each timestep is total reward
        ep_dict['reward'] = np.ones(ep_len) * rewards[i]
        
        
        for ax in range(len(ep_obs[0])):
            ep_dict[f'obs_ax{ax}'] = ep_obs[:, ax]

        ep_dict = pd.DataFrame(ep_dict)
        df_arr.append(ep_dict)
        
    #print(df_arr)
    #
    #import pdb; pdb.set_trace()
    
    df = pd.concat(df_arr)

    df.to_csv(filename)
    
    #heatmap(obs_matrix, "Observations by episode")
    
def seaborn_heatmap(df, xyz=('x','y','z'), annotate=False):
    print("Heatmap:")
    df.info()
    df[xyz[1]] = round(df[xyz[1]], 2)
    #cols = df.columns
    pivot = df.pivot_table(index = xyz[1], # rows
                                           values= xyz[2], # z-axis
                                           columns= xyz[0], 
                                           aggfunc=np.median)
    sb.heatmap(pivot, annot=annotate).set_title(f'xyz, {xyz}')
    
    plt.show()
    
    
    # sb == seaborn
    # sb.catplot() # categorical plot
    # ## Stack Figures
    # disp = sb.PairGrid(data)
    # disp.map(plt.scatter)
    
def plot_observation_reward(filename):
    df = pd.read_csv(filename)
    seaborn_heatmap(df, xyz=('timestep', 'obs_ax0', 'reward'))
    seaborn_heatmap(df, xyz=('timestep', 'obs_ax1', 'reward'))
    seaborn_heatmap(df, xyz=('timestep', 'obs_ax2', 'reward'))
    seaborn_heatmap(df, xyz=('timestep', 'obs_ax3', 'reward'))
    
def handle_cmdline():
    if len(argv) > 1:
        if argv[1] == "viz":
            # Visualize 50 episodes of random actions
            episodes(10, True)

        if argv[1] == "plot":
            # Plot 1000 obs and rewards as 4 heatmaps
            plot_csv = "plot_data.csv"
            save_observations(plot_csv)
            plot_observation_reward(plot_csv)

if __name__ == "__main__":
    handle_cmdline()
    
    filename = "random_ep_observation_rewards.csv"
    #save_observations(filename, epochs=100)
    #plot_observation_reward(filename)
    #episodes(1000, True)
    
    #plt.show()
