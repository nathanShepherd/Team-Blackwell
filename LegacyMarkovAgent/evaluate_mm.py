

import numpy as np
import matplotlib.pyplot as plt
#from collections import Counter
from markov_chain import MarkovAgent

import gym
env = gym.make('CartPole-v0')




def play_episode(epochs, agent, ep_length=200, viz=True):
  all_r = []
  for episode in range(epochs):
    observation = env.reset()
    ep_reward = 0
    f_reward = []
    ep_obs = []
    ep_acts = []
    
    for frame in range(ep_length):
      if viz:
        env.render()
      action = agent.act(observation)
      observation, reward, done, info = env.step(action)

      
      f_reward.append(ep_reward)
      ep_obs.append(observation)
      ep_acts.append(action)
      ep_reward += reward

      if done:
        print("\nEpisode", episode," completed after %s timesteps" % str(frame))
        f_reward[-1] = (0)
        agent.terminal(ep_obs, f_reward, ep_acts)
        all_r.append(ep_reward)
        break

  return all_r
      
def viz_performance(epochs, agent, ep_length=200, viz=True):
    all_r = []
    for episode in range(epochs):
      observation = env.reset()
      ep_reward = 0
      f_reward = []
      ep_obs = []
      ep_acts = []
    
      for frame in range(ep_length):
        if viz:
          env.render()
        action = agent.act(observation, frame, training=False)
        observation, reward, done, info = env.step(action)
      
        f_reward.append(ep_reward)
        ep_obs.append(observation)
        ep_acts.append(action)
        ep_reward += reward
      
        if done:
          print("\nEpisode", episode," completed after %s timesteps" % str(frame))
          print("Reward: ", ep_reward)
          f_reward[-1] = 0
          agent.terminal(ep_obs, f_reward, ep_acts)

          all_r.append(ep_reward)
          break
    return all_r
    
def main():  
  agent = MarkovAgent() #MarkovChain()
  epochs = 50
  rand_rew = play_episode(epochs, agent, viz=False)
  agent.view_model_params()
  agent_rew = viz_performance(epochs, agent, viz=True)
  agent.view_model_params()
  
  '''
  Todo: select highest reward sequence
  '''
  plt.plot(np.arange(epochs), rand_rew, label='Random')
  plt.plot(np.arange(epochs), agent_rew, label='Agent')
  plt.ylabel("Reward")
  plt.legend()
  plt.show()
  viz_performance(10, agent, viz=True)

if __name__ == "__main__":
  np.random.seed(0)
  main()
  #test_MM()

