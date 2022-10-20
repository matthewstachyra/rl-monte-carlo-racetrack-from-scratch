import numpy as np
import random

class Agent:
    def __init__(self, env):
        self.env = env
        self.s = self.env.get_start()
        self.x = self.s[0]
        self.y = self.s[1]
        self.hv = self.s[2]
        self.vv = self.s[3]
        self.a = [[0,0], [0,1], [1,0], [1,1], [-1,-1], [-1,0], [0,-1], [-1,1], [1,-1]]
        
    def get_actions(self):
        return self.a
    
    def get_start(self):
        self.s = self.env.get_start()
        return self.s
        
    def get_valid_actions(self):
        '''
        Actions are defined as the change in horizantal and vertical velocities. The changes
        can be 0, 1, or -1 to either or both of the horizantal and vertical velocities.
        
        Actions are valid for two conditions.
        (1) if the velocities aren't both 0 and neither exceeds 5.
        (2) if the position is within the bounds of the 2D environment.
        '''
        valid_actions = []
        xbound = self.env.get_bounds()[0]
        ybound = self.env.get_bounds()[1]
        for hv, vv in self.a:
            newy = self.y + vv
            newx = self.x + hv 
            newvv = self.vv + vv
            newhv = self.hv + hv
            if (newhv < 5) and (newhv >= 0) and (newvv < 5) and (newvv >= 0) and ~(newhv == 0 and newvv == 0):
                valid_actions.append([hv,vv])
        return valid_actions

    def update_agent(self, x, y, hv, vv):
        self.x = x
        self.y = y
        self.hv = hv
        self.vv = vv
        
    def reset_agent(self):
        self.s = self.env.get_start()
        self.x = self.s[0]
        self.y = self.s[1]
        self.hv = self.s[2]
        self.vv = self.s[3]
    
