'''
Entry point for program to create environment, agent, and on-policy monte carlo control.

Author: Matthew Stachyra
Date: 19 Oct 2022
'''

import os
import time
import random
from statistics import mean
import numpy as np

from .environment import Environment
from .agent import Agent
from .on-policy-mc-exploring-starts import OnPolicyMonteCarloControl as OPMC


t = "textbook-track.txt"
f = os.path.join(os.getcwd(), t)

env = Environment(f)
track = env.get_track()
a = Agent(env)
e = 0.1
d = 0.9
r = 1000

opmc_textbooktrack_testrun = OPMC(epsilon=e, 
        discount=d, 
        agent=a, 
        environment=env, 
        runs=r)
opmc_textbooktrack_testrun()
