# the base 
import numpy as np
import pandas as pd
import random
import os

import torch
import gym
import argparse

import config
from utils.wrappers import *
from utils.buffer_ import *
from utils.agent import *

def init_env(env_name, seed_):

    env = gym.make(env_name)

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = WarpFrame(env)
    env = PyTorchFrame(env)
    env = FrameStack(env, k=4)

    if not os.path.exists(config.video_path):
        os.mkdir(config.video_path)
    env = gym.wrappers.Monitor(
        env, './video/', video_callable=lambda episode_id: episode_id % 50 == 0, force=True)

    return env

def train(agent, env):

    eps_timesteps = config.eps_fra * float(config.num_steps)
    episode_rewards = [0.0]

    # reset the env
    state = env.reset()

    steps_ = []
    rewards_ = []
    episodes_ = []
    # begin training
    for t in range(config.num_steps):
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = config.eps_start + fraction * (config.eps_end - config.eps_start)
        sample = random.random()

        if(sample > eps_threshold):
            # Exploit
            action = agent.act(state)
        else:
            # Explore
            action = env.action_space.sample()
        
        next_state, reward, done, info = env.step(action)
        agent.memory.add(state, action, reward, next_state, float(done))
        state = next_state

        episode_rewards[-1] += reward
        # the game is over
        if done:
            state = env.reset()
            steps_.append(t)
            rewards_.append(episode_rewards[-1])
            episodes_.append(len(episode_rewards)-1)
            episode_rewards.append(0.0)
        
        # optimize the agent
        if t > config.learning_start and t % config.learning_freq == 0:
            agent.optimise_()

        # update the target network, which is interact with environment
        if t > config.learning_start and t % config.target_update_freq == 0:
            agent.update_target_network() 

        num_episodes = len(episode_rewards)-1

        if done and config.print_freq is not None and num_episodes % config.print_freq == 0:
            mean_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_reward))
            print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")
            torch.save(agent.policy_network.state_dict(), f'checkpoint.pth')
        
    result = pd.DataFrame()
    result['episode'] = episodes_
    result['step'] = steps_
    result['reward'] = rewards_

    return result

def main():
    # config.num_steps = int(2e4)
    # ------------- get the argument ------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-checkpoint', type = str, default=None,
                        help='input the path of the checkpoint file')

    args = parser.parse_args()
    if args.load_checkpoint:
        config.eps_start = 0.01
    else:
        config.eps_start = 1

    # ------------- initialization ------------------
    # set the random seed
    #np.random.seed(config.seed)
    #random.seed(config.seed)

    # init the environment
    env = init_env(config.env, config.seed)

    # init the buffer
    buffer_ = ReplayBuffer(config.buffer_size)

    # init the agent
    agent = DQNAgent(
        env.observation_space,
        env.action_space,
        buffer_,
        lr=config.learning_rate,
        batch_size=config.batch_size,
        gamma=config.discount_rate,
        device=torch.device("cuda:"+str(config.gpu_id) if torch.cuda.is_available() else "cpu")
    )

    # whether load the check point
    if args.load_checkpoint:
        print(f"Loading a policy - { args.load_checkpoint } ")
        agent.policy_network.load_state_dict(
            torch.load(args.load_checkpoint))
    
    # ------------- train ------------------
    res = train(agent, env)

    # ------------- save ------------------
    res.to_csv('result.csv')

 
if __name__ == '__main__':
    main()