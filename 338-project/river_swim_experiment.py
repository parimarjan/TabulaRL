'''
Script to run tabular experiments in batch mode.

author: iosband@stanford.edu
'''

import numpy as np
import pandas as pd
import argparse
import sys

import environment
import finite_tabular_agents

from feature_extractor import FeatureTrueState
from experiment import run_finite_tabular_experiment



if __name__ == '__main__':
    '''
    Run a tabular experiment according to command line arguments
    '''

    # Take in command line flags
    parser = argparse.ArgumentParser(description='Run tabular RL experiment')
    parser.add_argument('ep_len', help='length of episodes', type=int)
    parser.add_argument('num_states', help='number of states', type=int)
    parser.add_argument('alg', help='Agent constructor', type=str)
    parser.add_argument('scaling', help='scaling', type=float)
    parser.add_argument('seed', help='random seed', type=int)
    parser.add_argument('nEps', help='number of episodes', type=int)
    args = parser.parse_args()

    # Make a filename to identify flags
    fileName = ('chainLen'
                + '_len=' + '%03.f' % args.ep_len
                + '_num_states' + '%03.f' % args.num_states
                + '_alg=' + str(args.alg)
                + '_scal=' + '%03.2f' % args.scaling
                + '_seed=' + str(args.seed)
                + '.csv')

    folderName = './'
    targetPath = folderName + fileName
    print '******************************************************************'
    print fileName
    print '******************************************************************'

    # Make the environment
    # env = environment.make_riverSwim(args.ep_len, args.num_states)
    # env = environment.make_bootDQNChain(args.num_states, args.ep_len, 2)
    env = environment.make_stateBanditMDP(args.num_states, gap=0.1)

    # Make the feature extractor
    f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)

    # Make the agent
    alg_dict = {'PSRL': finite_tabular_agents.PSRL,
                'PSRLunif': finite_tabular_agents.PSRLunif,
                'OptimisticPSRL': finite_tabular_agents.OptimisticPSRL,
                'GaussianPSRL': finite_tabular_agents.GaussianPSRL,
                'UCBVI': finite_tabular_agents.UCBVI,
                'BEB': finite_tabular_agents.BEB,
                'BOLT': finite_tabular_agents.BOLT,
                'UCRL2': finite_tabular_agents.UCRL2,
                'UCFH': finite_tabular_agents.UCFH,
                'EpsilonGreedy': finite_tabular_agents.EpsilonGreedy}

    agent_constructor = alg_dict[args.alg]

    agent = agent_constructor(env.nState, env.nAction, env.epLen,
                              scaling=args.scaling)

    # Run the experiment
    run_finite_tabular_experiment(agent, env, f_ext, args.nEps, args.seed,
                        recFreq=100, fileFreq=1000, targetPath=targetPath)

