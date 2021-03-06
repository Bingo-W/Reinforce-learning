from gym import spaces
import numpy as np

from utils.model import DQN as DQN_neurips
from utils.buffer_ import ReplayBuffer
import torch
import torch.nn.functional as F


class DQNAgent:
    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Discrete,
                 replay_buffer: ReplayBuffer,
                 lr,
                 batch_size,
                 gamma,
                 device=torch.device("cpu" )):
        """
        Initialise the DQN algorithm using the Adam optimiser
        :param action_space: the action space of the environment
        :param observation_space: the state space of the environment
        :param replay_buffer: storage for experience replay
        :param lr: the learning rate for Adam
        :param batch_size: the batch size
        :param gamma: the discount factor
        """

        self.memory = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma
        DQN = DQN_neurips

        # get the policy network and target network
        self.policy_network = DQN(observation_space, action_space).to(device) 
        self.target_network = DQN(observation_space, action_space).to(device)
        
        self.update_target_network()
        # freeze the target network
        self.target_network.eval()

        self.optimiser = torch.optim.RMSprop(self.policy_network.parameters(), lr=lr)        
        ## self.optimiser = torch.optim.Adam(self.policy_network.parameters(), lr=lr)

        self.device = device

    def optimise_(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        device = self.device

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = np.array(states) / 255.0
        next_states = np.array(next_states) / 255.0

        # put the input into the gpu
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)

        with torch.no_grad():
            # get the max probably action
            _, max_next_action = self.policy_network(next_states).max(1) # get location of the max Q value, which is the optimal action
            max_next_q_values = self.target_network(next_states).gather(1, max_next_action.unsqueeze(1)).squeeze()
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        input_q_values = self.policy_network(states)
        input_q_values = input_q_values.gather(1, actions.unsqueeze(1)).squeeze()

        loss = F.smooth_l1_loss(input_q_values, target_q_values)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        del states
        del next_states
        return loss.item()

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def act(self, state: np.ndarray):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        device = self.device
        state = np.array(state) / 255.0
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.policy_network(state)
            _, action = q_values.max(1)
            return action.item()
