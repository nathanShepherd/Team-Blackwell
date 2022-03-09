import gym
import numpy as np
from collections import Counter

import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

# Use a Decision Tree Classifier as the Policy in CartPole
# Developed by Nathan Shepherd


actions = []# 1 or 0
obs = []# 4 by N vector
rewards = []# 1 at every timestep
'''
observation                 Min         Max
  0	Cart Position             -4.8            4.8
  1	Cart Velocity             -Inf            Inf
  2	Pole Angle                 -24           24
  3	Pole Velocity At Tip      -Inf            Inf
'''

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
        if episode % (epochs//5) == 0:
          print(f"\nEpisode: {episode}",
                    f"\tNum timesteps :{str(frame)}")
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
          if episode % (epochs//5) == 0:
            print(f"\nEpisode: {episode}",
                     f"\tNum timesteps :{str(frame)}")
          f_reward[-1] = 0
          agent.terminal(ep_obs, f_reward, ep_acts)

          all_r.append(ep_reward)
          break
    return all_r




  
class MarkovChain: 
  def __init__(self):
    self.decisions = [0, 0, 0, 0]

    self.observation_mem = []
    self.t_prob = None # Transition Probability
    self.e_prob = None # Emission Probability

  def fit_transition_prob(self, obs_by_time, n_states, prob='given'):
    # obs_by_time = states at time t in all obs
    # current_states = rows in table = n
    # next states = columns of table = m

    #import pdb; pdb.set_trace()

    local_table = np.zeros((n_states, n_states))
    
    if type(self.t_prob) == type(None):
      self.t_prob  = np.zeros((n_states, n_states))
    else:
      # Record current probs to update later
      local_table = self.t_prob
      
      # pad smaller transition table with zeros
      if n_states > len(self.t_prob[0]):
        pad_length = n_states - len(self.t_prob[0])
        self.t_prob = np.pad(self.t_prob, ((0, pad_length),( 0, pad_length)),
                                             constant_values=(0))
        local_table = self.t_prob
        
      elif  n_states < len(self.t_prob[0]):
        n_states = len(self.t_prob[0])
        
        
    current_states_count = np.zeros((n_states))

    # t_prob <-- P( current_state and next_state)
    for i, obs_t in enumerate(obs_by_time[:-1]):
      self.t_prob[obs_t][obs_by_time[i + 1]] += 1
      current_states_count[obs_t]  += 1

    # t_prob <-- P (s_(t + 1) | s_t) = P( current_state and next_state) / P(current_state)
    for state_idx in range(len(current_states_count)):
      if current_states_count[state_idx] != 0:
        self.t_prob[state_idx][:] /= current_states_count[state_idx]
    
    # Take the mean prob of prev estimate for t_prob and current prob
    self.t_prob = np.mean((self.t_prob.flatten() ,
                                             local_table.flatten() *5), axis = 0)
    
    self.t_prob = np.reshape(self.t_prob, (n_states, n_states))

    # Normalize mean probabilites to one
    for i in range(len(self.t_prob)):
      if prob == 'given':
        ssum =  sum(self.t_prob[:, i])
        row =  self.t_prob[:, i]
      elif prob == 'reverse':
        ssum =  sum(self.t_prob[i])
        row =  self.t_prob[i]
      
      if ssum != 0:
        row  /= ssum
      else:
        row = 0
        
      if prob == 'given':
        self.t_prob[:, i] = row 
      elif prob == 'reverse':
        self.t_prob[i] = row
        
      #print( self.t_prob[i], 'sum: ',   sum(self.t_prob[i]))

  def fit_emission_prob(self, obs, hidden, prob='given'):
    # obs_by_time = states at time t in all obs,
    #                            row for each of  emitted and emittor
    # current_states = rows in table = n hidden states
    # next states = columns of table = m emitted states

    hidden_counts = Counter(hidden)
    #print(hidden, obs)
    self.e_prob = np.zeros((max(hidden) + 1, max(obs) + 1))
    #print(self.e_table.shape)
    #print(obs.shape, hidden.shape)
    for t in range(0, len(obs)):
      h_idx = hidden[t]
      o_idx = obs[t]
      self.e_prob[h_idx][o_idx] += 1

    for hidden_state in hidden_counts:
      self.e_prob[hidden_state] /= hidden_counts[hidden_state]
    
    #print(self.e_prob)

    

      
  def prob_states_given_model(self, states):
    out_prob = 1
    for obs_t in range(len(states) - 1):
      out_prob *= self.t_prob[ states[obs_t] ][ states[obs_t + 1] ]
    return out_prob

  def expected_consecutive_obs(self, state):
    return 1 / (1.01 - self.t_prob[state][state])
    

  def proj_next_state_obs(self, current_obs):
    pass
    
  def act(self, obs, training=True):
    '''
    if training:
      # Only remember cart position
      
      obs = int(obs[2] * 10)
      self.observation_mem.append(obs)
    '''
    return np.random.randint(0, 2)

  def sigmoid_array(self, X):
    return 1 / (1 + np.exp( -X))
  
  def terminal(self, obs, reward):
    # Only observe cart position
    cleaned_obs = self.sigmoid_array(np.array(obs)[:,2])
    
    print('{Terminal obs: {', cleaned_obs, '}, reward: {', reward, '}')
      
    #self.act(obs)
    
    self.observation_mem = list(map(lambda x: int( round(x, 0)), cleaned_obs))
    print("Model's Observation_Memory: \n", self.observation_mem)
    
    self.fit_transition_prob(self.observation_mem,
                                                  max(self.observation_mem) + 1)

    #self.view_model_params()

  def view_model_params(self):
    print("Model Parameters:")
    print("=============")
    print("Transition Table:")
    for i, row in enumerate(self.t_prob):
      if i == 0:  
        pass #print(len(self.t_prob[row]))
      row = list(map(lambda x: round(x, 1), row))
      print("P( curr_obs = ", i, " | next_obs  = col )", row)
      
    for state in range(len(self.t_prob)):
      E_i = self.expected_consecutive_obs(state)
      print(f'State: {state}, E[observing {state}]: {E_i}')

def test_MM():
  agent = MarkovChain()

  obs_by_time = [1, 1, 1, 0]
  num_states = 2

  #import pdb; pdb.set_trace()
  
  agent.fit_transition_prob(obs_by_time, num_states)
  
  print("Observations", obs_by_time)
  print("Transition Table:\n", agent.t_prob)
  
  for state in range(num_states):
    E_i = agent.expected_consecutive_obs(state)
    print('State: {', state, '}, E[observing {state}]: {', E_i, '}')

'''
Agent combines multiple fully observed Markov Chains
'''
class MarkovAgent():
  def __init__(self):
    self.obs_to_obs = MarkovChain()
    self.act_to_act = MarkovChain()
    
    self.act_obs = MarkovChain()
    self.obs_act = MarkovChain()
    self.reward_obs = MarkovChain() # Reward given Observation

    self.obs_expected_reward = None

    self.best_acts = []
    
    #self.rewards = np.array([])
    self.frame_actions = []
    self.all_actions = []
    self.all_obs = []
    self.all_rewards = []

  def act(self, obs, time_t =0, training=True):
    if training:

      action = np.random.randint(0, 2)
      self.all_actions.append(action)
      return action
    else:
      if time_t < len(self.best_acts) < 0.5:
        action = self.best_acts[time_t]
        self.all_actions.append(action)
        return action
      
    
      # Skip Below Algorithm if best action so far has been seen
      action_reward = []
      
      for action in range(2):
        # Most likely next observation given this action
        most_likely_obs = np.random.choice(np.arange(0, len(self.act_obs.e_prob[0])),
                                                                           p=self.act_obs.e_prob[action])
        '''
        expected_obs = 0
        for col in range(len(self.act_obs.e_prob[action])):
          expected_obs += col * self.act_obs.e_prob[action][col]
        most_likely_obs = int(expected_obs)
        '''
        #print('most_likely_obs', most_likely_obs)
        
        # Transition one timestep
        #most_likely_obs = np.random.choice(np.arange(0, len(self.obs_to_obs.t_prob[most_likely_obs])),
         #                                                                  p=self.obs_to_obs.t_prob[most_likely_obs])
        #print('most_likely_transition', most_likely_obs)
        
        # Expected reward of this observation given this action
        #action_reward.append(self.obs_expected_reward[most_likely_obs])
        #import pdb; pdb.set_trace()
        
        #max_expected = np.argmax(self.reward_obs.e_prob[most_likely_obs])
        
        expected_rew = np.random.choice(np.arange(0, len(self.reward_obs.e_prob[most_likely_obs])),
                                                                      p=self.reward_obs.e_prob[most_likely_obs])
        expected_rew *= self.reward_obs.e_prob[most_likely_obs][-1]
        
        #for col in range(len(self.reward_obs.e_prob[most_likely_obs])):
         # expected_rew += (col + 1) * self.reward_obs.e_prob[most_likely_obs][col]
        
        if action == self.all_actions[-1]:
          expected_rew *= 0.8

        # Add greater weighting to obs to obs transitions that occur less often
        #max_expected *= self.obs_to_obs.t_prob[most_likely_obs][most_likely_obs]
        action_reward.append(expected_rew)
        
        '''
        expected_rew = 0
        for col in range(len(self.reward_obs.e_prob[most_likely_obs])):
          expected_rew += (col + 1) * self.reward_obs.e_prob[most_likely_obs][col]
        action_reward.append(expected_rew)
        '''

      #print('Act:E[Rew] ', action_reward)
      #action = int(np.argmax(action_reward))
      action_reward = np.array(action_reward)
      if sum(action_reward) == 0:
          if self.all_actions[-1] == 1:
            action = 0
          else:
            action = 1
      else:
        action = np.random.choice(np.arange(0, len(action_reward)), p=action_reward/ sum(action_reward))
      self.all_actions.append(action)

      #if np.random.randint(0, 11) < 1:
      # action = np.random.randint(0, 2)
      return action
      '''
      if type(self.obs_act.t_prob) == type(None):
        return np.random.randint(0, 2)
      else:
        # TODO: Determine action with greatest reward
        cleaned_obs = self.sigmoid_array(np.array([obs[2]]))[0]
        cleaned_obs = int( round(cleaned_obs, 0))
        next_state = np.argmax(self.obs_to_obs.t_prob[cleaned_obs])
        act_taken = np.argmax(self.obs_act.t_prob[next_state])
        #act_taken = np.argmax(self.act_reward.t_prob[:,-1])
        return act_taken
      '''
      
  def sigmoid_array(self, X, num_bins = 20, upper_lim = 5):
    #num_bins = 20 # adjust until observed in range(0, upper_lim)
    #upper_lim = 5 # for greater size ints
    # 10 bin = (z-100, ..., z0, ... z100 --> (01, ..., x5, x6, x7))
    # which is equal to 7 action bins in the range(-100, 100)
    # where z10 == 10, x5 == 0.5
    X = upper_lim / (1 + np.exp( -num_bins*X))
    #print(X)
    X = list(map(lambda x: int( round(x, 3)), X))
    #print(X)
    
     
    return X
    
  
    
  def terminal(self, obs, reward, acts=[]):
    '''
    observation                 Min         Max
    0	Cart Position             -4.8            4.8
    1	Cart Velocity             -Inf            Inf
    2	Pole Angle                 -24           24
    3	Pole Velocity At Tip      -Inf            Inf
    '''
    if len(acts) > 0 and len(acts)/2 > len(self.best_acts) / 3 and len(self.best_acts) < len(acts):
      print("replacing best actions", len(self.best_acts), '-->', len(acts))
      self.best_acts = acts
      
    cleaned_obs = self.sigmoid_array(np.array(obs)[:,0])

    #import pdb; pdb.set_trace()
    '''
    print(reward, self.running_mean_rew)
    
    if len(self.all_rewards) != 0 and np.mean(reward) < self.running_mean_rew:
      return 0
    self.running_mean_rew = (sum(reward)) / len(reward)
    '''
    #print('{Terminal obs: {', cleaned_obs, '}, Terminal reward: {', reward, '}')
    self.obs_to_obs.observation_mem = cleaned_obs
    #print("Obs: ", self.obs_to_obs.observation_mem)
    self.all_obs.append(cleaned_obs)
    
    self.obs_to_obs.fit_transition_prob(self.obs_to_obs.observation_mem,
                                                  max(self.obs_to_obs.observation_mem) + 1,  'reverse')
    #print(reward)
    reward = list(map(lambda x: int( x), reward))
    #reward = self.sigmoid_array(np.array(reward), 100, 10)
    #print('Reward: ', self.all_rewards, reward)
    reward = np.array(reward)
    self.all_rewards.append(reward)

    loc_acts = np.array(self.all_actions)
    loc_obs = np.concatenate(self.all_obs)
    loc_rewards = np.concatenate(self.all_rewards)

    self.reward_obs.fit_emission_prob(loc_rewards, loc_obs)
    #print(loc_obs.shape, loc_acts.shape)
    self.act_obs.fit_emission_prob(loc_obs, loc_acts)
    self.act_to_act.fit_transition_prob(loc_acts, max(loc_acts) + 1)
    
    #self.rewards = np.concatenate((self.rewards, reward))
    #print("Act:  ", self.actions)
    #print('Reward: ', self.rewards)
    '''
          TODO: assess different bin sizes
                        combine information from each row intelligently
                        create obs_rew.t_table rows sum to one
    '''
    
    '''
    obs_by_actions = np.array(list(zip( self.all_actions,
                                        self.obs_to_obs.observation_mem,))).astype(int)
                                                                 
    obs_by_reward = np.array(list(zip(self.obs_to_obs.observation_mem,
                                                                  self.rewards,))).astype(int)
    #print('obs_by_actions_reward', obs_by_actions_reward)
    for row in obs_by_actions:
          
          upper_limit = max(row) + 1        
          self.act_obs.fit_transition_prob(row, upper_limit, 'reverse')

    for row in obs_by_reward:
          upper_limit = max(row) + 1
          self.obs_reward.fit_transition_prob(row, upper_limit,)

    # Expected Reward given an observation
    self.obs_expected_reward = []
    for row in self.obs_reward.t_prob:
      for col in range(len(row)):
        row[col] = col * row[col]
      self.obs_expected_reward.append(sum(row))
    '''
    '''
    for col in range(len(obs_by_actions_reward[0]) -1 ):
      if col == 0:
        part = obs_by_actions_reward[:, col:col + 2]
        #part_reverse = np.flip(obs_by_actions_reward[:, col:col + 2], axis =1)
        #print('act/obs: ', part) ###
        

        # Should not be the same?
        #for row in part_reverse:
        #  self.obs_act.fit_transition_prob(np.flip(row), upper_limit)
          
      elif col == 1:
        # remove actions
        del obs_by_actions_reward[:, 1] 
        part = obs_by_actions_reward
        #print("obs/rew", part) ###
    '''  
        

    #self.actions = []
    

  def view_model_params(self):
    

    #print("E[ Reward | Obs]")
    #print(self.obs_expected_reward)

    n_a = np.array 
    print(f"All Rewards\n{n_a(self.all_rewards)}")
    print(f"All Observations\n {n_a(self.all_obs)}")
    print(f"All Actions\n {n_a(self.all_actions)}")

    print("\nP(Reward |  Observation), row=Given")
    print(self.reward_obs.e_prob)

    print("\nP(Observation |  Action), row=Given")
    print(self.act_obs.e_prob)
    
    print("\nrow, col --> Obs, Obs")
    self.obs_to_obs.view_model_params()

    print("\nrow, col --> Act, Act")
    self.act_to_act.view_model_params()
    
    #print("\nrow = col --> Obs = Act")
    #self.act_obs.view_model_params()
    
def main():  
  agent = MarkovAgent() #MarkovChain()
  epochs = 10
  rand_rew = play_episode(epochs, agent, viz=False)
  agent.view_model_params()
  agent_rew = viz_performance(epochs, agent, viz=False)
  agent.view_model_params()
  '''
  Todo: select highest reward sequence
  '''
  plt.plot(np.arange(epochs), rand_rew, label='Random')
  plt.plot(np.arange(epochs), agent_rew, label='Agent')
  plt.ylabel("Reward")
  plt.legend()
  plt.show()
  viz_performance(epochs, agent, viz=True)

if __name__ == "__main__":
  np.random.seed(0)
  main()
  #test_MM()
