import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

'''
P(A) = num_A / total
         =  num_A / P(A and B)

P(A | B) = P(A and B) / P(A)
                 = num_A / (total after select A)
                 = proportion of A in B

P(A and B) = P(A) * P(B | A)
                    = select A, then B given A
                    
         if independant:
                    = P(A) * P(B)
                    = select A and B
'''

def P(value, state, given=None, dependant=True):
    count = Counter(state)
    p = 0
    
    if given == None:
         p = count[value] / len(state)
         
    elif dependant:
        # Prob of value given the value of given
        # P(value and given) / P(value)
        # --> (num_value / a
        p = count[value] / len(state)
        p *= Counter(given)[value]
        
    else:
        P = P(value, state) * P(given, state)
        
    return p
'''
a = [1, 1, 0, 1, 0]
b = [0, 0, 0, 1, 0]
state = np.concatenate([a, b])
'''
#print(state)
#print(P(True, state, given = a))

  
def generator(timesteps, table_rows=1):
    ''' Generate table_rows number of random samples '''
    ''' Each of y_values has a prob of appearing in a sample '''
    # timesteps is the number of samples per observation ( len(obs) )
    
    # table_rows is number of containers to put y's
    # y_length is the number of  possible y values
    # y_values are possible values of y
    y_length = 2
    y_values = [True, False] 
    

    y0_prob = 0.8
    y1_prob = 0.2
    prob = [y0_prob, y1_prob]

    
    
    hidden_obs = [] # record of sample that generates prob_table

    # iff table_rows = 1

    # generate data to compute prob of sample
    hidden_obs = np.random.choice(y_values,
                                                    p=prob, replace=True,
                                                    size=(timesteps))
        
    # end iff
    '''
    for x in range(table_rows): # Populate prob_table

        # generate data to compute prob of sample
        observations = np.random.choice(y_values,
                                                    p=prob, replace=True,
                                                    size=(timesteps))
        hidden_obs.append(observations)
    '''
    #print('gen obs', hidden_obs)
    return hidden_obs
    
class MarkovModel:
    def __init__(self):
        pass

    def start_value_prob(self, initial_states):
        '''
        Note: start_value_prob can be calculated from trans_prob
        
            P(True) = P(TrueTrue) + P(FalseTrue)
            P(False) = P(FalseFalse) + P(TrueFalse)
            1 = P(True) + P(False)
            
            solve for P(True) and P(False)
        '''
        #print('start', initial_states)
        
        total_obs = len(initial_states)
        count = Counter(initial_states)

        for y_value in count:
            count[y_value] /= total_obs
        count = dict(count)
        
        print('start prob', count)
        self.start_prob = count # TODO: update with uncertainty of state
        return count


    def transition_prob(self, hidden_obs):
       # print('hidden_obs', hidden_obs)
        unique = np.unique( hidden_obs)
        #print('unique', unique)

        cond_dict = {} # dictionary of prob of y_val given the prev y_val
        count_unique = {}

        # initialize cond_dict with possible sequences of y_val
        # using single order sequence
        for first_y in unique:
            first_y = str(first_y)
            count_unique[first_y] = 0
            
            for second_y in unique:
                key = first_y + str(second_y)
                cond_dict[key] = 0

        # update counts for each dependant value in hidden_obs
        #                   and for each unique sequence
        for col_idx in range(len(hidden_obs) - 1):
            
            dependant = str(hidden_obs[col_idx])
            #print('dependant', dependant)
            # count of this valued observation given the dependant (prev)
            given_dep = dependant + str(hidden_obs[col_idx + 1])

            count_unique[dependant] += 1
            cond_dict[given_dep] += 1
            
        #print('unique', count_unique)
        #print('conditional', cond_dict)
        
        # Divide each conditional count by the total of each dependant event
        
        for dependant, count_dep in count_unique.items():
            #print(dependant, count_dep)
            # select all key:value from cond_dict of substr (dep) in key
            filter_substr = lambda item: dependant == item[0][:len(dependant)]
            matched_keys = dict(filter(filter_substr, cond_dict.items()))

            #print('matched', matched_keys)
            for key in matched_keys:
                cond_dict[key] /= count_dep
                
        print('hidden prob given hidden', cond_dict)
        self.p_h_given_h = cond_dict # TODO: update with uncertainty of state
        return cond_dict
                

    def emission_prob(self, X, hidden_obs):
        prob_x_given_h = {}

        # initialize dict conditional probability of x given hidden state
        unique_h = np.unique(hidden_obs)
        unique_x = np.unique(X)
        for hidden_val in unique_h:
            prob_x_given_h[hidden_val] = {}
            for x_val in unique_x:
                prob_x_given_h[hidden_val][x_val] = 0

        # calculate count of x_given_h
        for idx in range(len(X)):
            prob_x_given_h[hidden_obs[idx]][X[idx]] += 1

        # calculate prob_x_given_h from counts for x_given_h
        for hidden_val in unique_h:
            total_x = sum(prob_x_given_h[hidden_val].values())
            #print(hidden_val, total_x)

            for x_prob in unique_x:
                prob_x_given_h[hidden_val][x_prob] /= total_x

        print('x given h', prob_x_given_h)
        #p (x | h ) --> {False: {0: 0.5, 1: 0.5}, True: {0: 0.0, 1: 1.0}}
        
        self.p_x_given_h = prob_x_given_h # TODO: update with uncertainty of state
        return prob_x_given_h

    def viterbi_algo(self, X):

        # Choose maximum starting prob
        max_likely_seq = [] # most likely sequence of hidden state given X
        
        prev_max_h = {}
        for hidden_val in self.p_x_given_h:
            prob = self.p_x_given_h[hidden_val][X[0]] * self.start_prob[hidden_val] 
            prev_max_h[hidden_val] =  prob
            
        max_key = max(prev_max_h, key=prev_max_h.get)
        max_likely_seq.append(max_key)
        
        #print(prev_max_h)
        #print('Max Likely h_state:', max_key, '; with prob:', prev_max_h[max_key])

        for x_obs in X[1:]:
            new_max_prob = {}
            for current_h_val in self.p_x_given_h:
                max_prob_of_current_h_val = []
                
                for prev_h_val in self.p_x_given_h:
                    # max probability of observing previous states given observations of x
                    prob = prev_max_h[prev_h_val]

                    # probability of observing x given hidden state
                    prob *= self.p_x_given_h[current_h_val][x_obs]

                    # probabilty of transition
                    condition_prev_h = str(prev_h_val) + str(current_h_val)
                    prob *= self.p_h_given_h[condition_prev_h]
                    
                    max_prob_of_current_h_val.append(prob)
                    
                new_max_prob[current_h_val] =  max(max_prob_of_current_h_val)

            prev_max_h = new_max_prob
            
            max_key = max(prev_max_h, key=prev_max_h.get)
            max_likely_seq.append(max_key)

        print("max likely sequence", max_likely_seq)
        return max_likely_seq

    def baum_welch_algo(X):
        pass
        
            
if __name__ == "__main__":
    np.random.seed(5)
    
    #for i in range(5):
    #   print(generator())

    #print(generator(5))

    # Generator emits True or False for given num of timesteps
    #   P(T, F) predefined --> P(T) == 0.8, P(F) == 0.2
    obs_table = generator(3)
    print(obs_table) # --> [True, False, True]
    
    
    MM = MarkovModel()
    MM.start_value_prob([True, False, True])
    MM.transition_prob(obs_table)
    MM.emission_prob([1, 0, 1], [True, False, False])

    MM.viterbi_algo([1, 0, 1])
    

















    

    
        
