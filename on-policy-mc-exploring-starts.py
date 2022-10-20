import os
import time
import random
from statistics import mean
import numpy as np

class OnPolicyMonteCarloControl:
    '''
    Behavior policy is e-greedy with epsilon 0.1.
    Target policy is greedy.
    Exploring starts (i.e., different start position for each generation episode, for each run).
    '''
    
    def __init__(self, epsilon, discount, agent, environment, runs):
        self.epsilon = epsilon
        self.discount = discount
        self.agent = agent
        self.env = environment
        self.runs = runs
        self.episode = {}               # e[t] = (s, r), where s=6-tuple(x,y,hv,vv,h,v), r=reward
        self.Q = self.initialize("q")   # Q[s] = q, where q=value
        self.Gs = self.initialize("g")  # Gs[s] = G or return through this s, a pair
                    
    def reset(self):
        self.agent.reset_agent()
        self.episode = {}
        
    def get_episode(self):
        return self.episode
        
    def initialize(self, which):
        d = {}
        hvelocities = vvelocities = [0,1,2,3,4]
        for x in range(len(self.env.get_track())):
            for y in range(len(self.env.get_track()[x])):
                for hv in hvelocities:
                    for vv in vvelocities:
                        s = [x, y, hv, vv]
                        for a in self.agent.get_actions():
                            sprime = s.copy()
                            sprime.extend(a)
                            sprime = tuple(sprime)
                            if which=="q":
                                d[sprime] = 0
                            if which=="g":
                                d[sprime] = [] 
        return d

    def generate_episode(self):
        self.agent.reset_agent()
        s = self.agent.get_start() # [x, y, hv, vv]
        s.extend([0,0])            # [x, y, hv, vv, h, v]
        
        x = s[0]
        y = s[1]
        hv = s[2]
        vv = s[3]
        
        t = 0
        r = 0
        tr = r
        oob = 0
        
        while (self.env.check_not_finished(x, y)):
            
            # first take the action that was stored in 
            # the tuple (which is (0,0) to start)
            scopy = list(s).copy()
            hv = scopy[2] + scopy[4] # horizantal speed
            vv = scopy[3] + scopy[5] # vertical speed
            x = scopy[0] - vv  # horizantal position
            y = scopy[1] + hv  # vertical position

            # check out of bounds / update reward
            if self.env.check_out_of_bounds(x, y):
                s = self.env.get_start() # [x, y, hv, vv]
                x, y, hv, vv = s[0], s[1], 0, 0
                r = -100 # overwrite reward because out of bounds
                oob += 1
            else:
                r = self.env.get_reward(x, y)
            
            # update episode with old values (i.e., the reward is for t+1 and the state is that at t)
            self.episode[t] = ((scopy[0], scopy[1], scopy[2], scopy[3], scopy[4], scopy[5]), r)

            
            # then, get next action
            self.agent.update_agent(x, y, hv, vv)
            s_without_actions = [x, y, hv, vv]              
            a = self.get_next_action(s_without_actions) # [h, v]
            
            # with 10% probability, the velocity increments are both 0 at each time step
            if random.random() < 0.1:
                a = [0,0]

            # save to new state
            s = tuple([x, y, hv, vv, a[0], a[1]])
            
            # update time
            t += 1
            
            # track total reward for episode
            tr += r
            
        
        # append last episode (finish state), with reward 0
        self.episode[t] = ((x, y, hv, vv, a[0], a[1]), 0)
        print(f"Finished in {t} steps with {tr} reward accumulated and went out of bounds {oob} times.")
        print()
            
        
    def get_next_action(self, state):
        '''state has form (x, y, hv, vv)
        '''
        if(np.random.binomial(1, self.epsilon)): # 1 if exploring, 0 if exploiting
            return self.agent.get_valid_actions()[random.randint(0, len(self.agent.get_valid_actions())-1)]
        return self.get_greedy_choice(state)
    
    def get_greedy_choice(self, state):
        '''state has form (x, y, hv, vv)
        '''
        valid_actions = self.agent.get_valid_actions()
        sas = [] # list of (x, y, hv, vv, h, v)
        for va in valid_actions:
            s = state.copy()
            s.extend(va)          # add new action
            sas.append(tuple(s))  # make to tuple (x,y,hv,vv,h,v)
            
        # if all q values are 0, then we don't have action values for the (s,a), so return random
        if sum([self.Q[sa] for sa in sas])==0: return valid_actions[random.randint(0, len(valid_actions)-1)]
        # else, check least negative / largest q value and return that action
        ma = [0,0]
        mq = -11111111
        for sa in sas:
            if self.Q[sa] > mq:
                mq = self.Q[sa]
                ma = sa[-2:]
        return ma

    def __call__(self):
        '''
        On-Policy Monte Carlo Exploring Starts
        '''
        for nr in range(self.runs):
            self.reset()
            self.generate_episode()
            G = 0
            T = len(self.episode)
            visited = []
            for t in range(T - 1, -1, -1): 
                step = self.episode[t]  # step is (s,r) or ((x,y,hv,vv,h,v), r)
                sa = step[0]
                s = sa[:-2] # first 4 items in 6-tuple, which is (x,y,hv,vv)
                a = sa[-2:] # last 2 items in 6-tuple, which is (h,v)
                r = step[1]
                G = self.discount*G + r
                if s not in visited:
                    self.Gs[sa].append(G)
                    self.Q[sa] = mean(self.Gs[sa])
                visited.append((s,a))
