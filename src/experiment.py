'''
Script to run simple tabular RL experiments.

author: iosband@stanford.edu
'''

import numpy as np
import pandas as pd
from shutil import copyfile

def run_finite_tabular_experiment(agent, env, f_ext, nEps, seed=1,
                    recFreq=100, fileFreq=1000, targetPath='tmp.csv'):
    '''
    A simple script to run a finite tabular MDP experiment

    Args:
        agent - finite tabular agent
        env - finite TabularMD
        f_ext - trivial FeatureTrueState
        nEps - number of episodes to run
        seed - numpy random seed
        recFreq - how many episodes between logging
        fileFreq - how many episodes between writing file
        targetPath - where to write the csv

    Returns:
        NULL - data is output to targetPath as csv file
    '''
    cumRegrets = []
    data = []
    qVals, qMax = env.compute_qVals()
    np.random.seed(seed)

    cumRegret = 0
    cumReward = 0
    empRegret = 0
    env.reset()

    for ep in xrange(1, nEps + 2):
        # Reset the environment
        env.reset()
        epMaxVal = qMax[env.timestep][env.state]

        agent.update_policy(ep)

        epReward = 0
        epRegret = 0
        pContinue = 1

        while pContinue > 0:
            # Step through the episode
            h, oldState = f_ext.get_feat(env)

            action = agent.pick_action(oldState, h)
            epRegret += qVals[oldState, h].max() - qVals[oldState, h][action]

            reward, newState, pContinue = env.advance(action)
            epReward += reward

            agent.update_obs(oldState, action, reward, newState, pContinue, h)

        cumReward += epReward
        cumRegret += epRegret
        empRegret += (epMaxVal - epReward)

        # Variable granularity
        if ep < 1e4:
            recFreq = 100
        elif ep < 1e5:
            recFreq = 1000
        else:
            recFreq = 10000

        # if epReward > 0:
            # # pari: sometimes regret is negative, why? is epMaxVal expected max
            # # and not actual max?
            # print('episode: {}, reward: {}, regret: {}'.format(ep, epReward,
                # epMaxVal-epReward))
        # else:
            # print('reward is 0')
            # print('max reward is: ', epMaxVal)

        # Logging to dataframe
        if ep % recFreq == 0:
            cumRegrets.append(cumRegret)
            # data.append([ep, epReward, cumReward, cumRegret, empRegret])
            # print 'episode:', ep, 'epReward:', epReward, 'cumRegret:', cumRegret

        # if ep % max(fileFreq, recFreq) == 0:
            # dt = pd.DataFrame(data,
                              # columns=['episode', 'epReward', 'cumReward',
                                       # 'cumRegret', 'empRegret'])
            # print 'Writing to file ' + targetPath
            # dt.to_csv('tmp.csv', index=False, float_format='%.2f')
            # copyfile('tmp.csv', targetPath)
            # print '****************************'

    return cumRegrets
    print '**************************************************'
    print 'Experiment complete'
    print '**************************************************'


def run_random_search_experiment(agent, env, f_ext, nEps, seed=1,
                    recFreq=100, fileFreq=1000, targetPath='tmp.csv'):
    def run_episode():

        epReward = 0
        epRegret = 0
        pContinue = 1

        while pContinue > 0:
            # Step through the episode
            h, oldState = f_ext.get_feat(env)

            action = agent.pick_action(oldState, h)
            epRegret += qVals[oldState, h].max() - qVals[oldState, h][action]

            reward, newState, pContinue = env.advance(action)
            epReward += reward

            # FIXME: do we need to do anything here?
            # agent.update_obs(oldState, action, reward, newState, pContinue, h)

        return epReward, epRegret

    env.reset()
    cumRegrets = []
    data = []
    qVals, qMax = env.compute_qVals()
    np.random.seed(seed)

    cumRegret = 0
    cumReward = 0
    empRegret = 0

    for batch in xrange(0, ((nEps + 1) / (agent.batch_size*2))):
        # Reset the environment
        epMaxVal = qMax[env.timestep][env.state]
        reward_differences = 0

        #create list to store all the rewards
        list_of_pos_rewards = []
        list_of_neg_rewards = []
        list_of_noise = []


        for ep in xrange(agent.batch_size):

            noise = np.random.normal(size=agent.theta.shape)
            theta_pos = agent.theta + (noise*agent.v)
            theta_neg = agent.theta - (noise*agent.v)

            env.reset()
            agent.cur_theta = theta_pos
            epRewardPos, epRegretPlus = run_episode()

            ## negative batch
            env.reset()
            agent.cur_theta = theta_neg
            epRewardNeg, epRegretMinus = run_episode()

            list_of_pos_rewards.append(epRewardPos)
            list_of_neg_rewards.append(epRewardNeg)
            list_of_noise.append(noise)

            ## update reward differences.
            #reward_differences += (epRewardPos - epRewardNeg)*noise

            cumReward += epRewardPos + epRewardNeg
            cumRegret += epRegretPlus + epRegretMinus
            empRegret += (epMaxVal - epRewardPos) + (epMaxVal - epRewardNeg)

            # Variable granularity
            cur_ep = batch*agent.batch_size*2 + ep*2

            if cur_ep < 1e4:
                recFreq = 100
            elif cur_ep < 1e5:
                recFreq = 1000
            else:
                recFreq = 10000

            # Logging to dataframe

            # FIXME: how often do we want it to record.
            if cur_ep % recFreq == 0:
                # data.append([cur_ep, epRewardPos, cumReward, cumRegret, empRegret])
                # print 'episode:', cur_ep, 'epRewardPos:', epRewardPos, 'cumRegret:', cumRegret
                cumRegrets.append(cumRegret)

            # if cur_ep % max(fileFreq, recFreq) == 0:
                # dt = pd.DataFrame(data,
                                  # columns=['episode', 'epReward', 'cumReward',
                                           # 'cumRegret', 'empRegret'])
                # print 'Writing to file ' + targetPath
                # dt.to_csv('tmp.csv', index=False, float_format='%.2f')
                # copyfile('tmp.csv', targetPath)
                # print '****************************'


        list_of_pos_rewards = np.array(list_of_pos_rewards)
        list_of_neg_rewards = np.array(list_of_neg_rewards)
        list_of_noise= np.array(list_of_noise)

        reward_differences = 0
        for i in range(0, len(list_of_pos_rewards)):
            reward_differences += (list_of_pos_rewards[i] - list_of_neg_rewards[i])*list_of_noise[i]
        

        #reward_differences = np.sum((list_of_pos_rewards - list_of_neg_rewards)*list_of_noise)

        if (agent.version == 'v1'):
            sigma_R = np.std(np.append(list_of_pos_rewards, list_of_neg_rewards))
            if sigma_R == 0:
                sigma_R = 1
            agent.theta = agent.theta + (agent.alpha / (agent.batch_size*sigma_R))*reward_differences

        if (agent.version == 'v1-t'):
            maxlist = []
            for i in range(0, len(list_of_pos_rewards)):
                maxlist.append(max(list_of_pos_rewards[i], list_of_neg_rewards[i]))

            maxlist = np.array(maxlist)
            top_b_indices = maxlist.argsort()[-agent.b:][::-1]

            top_pos = list_of_pos_rewards[top_b_indices]
            top_neg = list_of_neg_rewards[top_b_indices]
            top_noise = list_of_noise[top_b_indices]

            reward_differences = 0
            for i in range(0, len(top_pos)):
                reward_differences += (top_pos[i] - top_neg[i])*top_noise[i]


            sigma_R = 1
            #sigma_R = np.std(np.append(top_pos, top_neg))
            #if sigma_R == 0:
                #sigma_R = 1
            agent.theta = agent.theta + (agent.alpha / (agent.batch_size*sigma_R))*reward_differences





        else: agent.theta = agent.theta + (agent.alpha / agent.batch_size)*reward_differences

    return cumRegrets

    print '**************************************************'
    print 'Experiment complete'
    print '**************************************************'


