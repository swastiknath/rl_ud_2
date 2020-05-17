from collections import deque, namedtuple
import numpy as np
import random
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model import Actor, Critic

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 2e-4
LR_CRITIC = 2e-4
WEIGHT_DECAY = 0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ExperienceReplayBuffer():
    '''
    EXPERIENCE REPLAY BUFFER.
    '''
    def __init__(self, action_size, buffer_size, batch_size, seed=0):
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.memory = deque(maxlen=buffer_size)
        
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def __len__(self):
        return len(self.memory)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)
    
class OUNoiseProcess:
    
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1, parallal=False):
        self.mu = mu*np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.seed = random.seed(seed)
        self.parallal=parallal
        self.reset()
        
        
    def reset(self):
        self.state = copy.copy(self.mu)
        
    def sample(self):
        x = self.state
        if self.parallal:
            dx = self.theta * (self.mu - x) + self.sigma * np.random.normal(0, 1, self.size)
        else:
            dx = self.theta * (self.mu  - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state



class Agent():
    
    def __init__(self, state_size, action_size, random_seed=0, parallal=False, num_agents=0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.parallal=parallal
        self.num_agents=num_agents
        self.actor_online = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_online.parameters(), lr=LR_ACTOR, weight_decay=WEIGHT_DECAY)
        
        self.critic_online = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic_online.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        self.copy_weights(self.actor_target, self.actor_online)
        self.copy_weights(self.critic_target, self.critic_online)
        
        if self.parallal:
            self.noise = OUNoiseProcess((num_agents, action_size), random_seed, parallal=self.parallal)
        else:
            self.noise = OUNoiseProcess(action_size, random_seed)
        
        self.memory = ExperienceReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
    
    def copy_weights(self, target, online):
        for target_param, online_param in zip(target.parameters(), online.parameters()):
            target_param.data.copy_(online_param.data)
            
    def step(self, state, action, reward, next_state, done):
        
        if self.parallal:
            for i in range(self.num_agents):
                self.memory.add(state[i,:], action[i,:], reward[i], next_state[i,:], done[i])

        else:
            self.memory.add(state, action, reward, next_state, done)
            
        if len(self.memory) >  BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
            
    def act(self, state, noisy=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_online.eval()
        with torch.no_grad():
            if self.parallal:
                action = np.zeros((self.num_agents, self.action_size))
                for num, state_ in enumerate(state):
                    action[num,:] = self.actor_online(state_).cpu().data.numpy() 
            else:
                action = self.actor_online(state).cpu().data.numpy()
        self.actor_online.train()
        if noisy:
            action += self.noise.sample()
        return np.clip(action, -1, 1)
    
    def reset(self):
        self.noise.reset()
        
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1-dones))
        
        Q_expected = self.critic_online(states, actions)
        
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_online.parameters(), 1)
        self.critic_optimizer.step()
        
        
        actions_pred = self.actor_online(states)
        actor_loss = -self.critic_online(states, actions_pred).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.soft_update(self.critic_online, self.critic_target, TAU)
        self.soft_update(self.actor_online, self.actor_target, TAU)
        

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
            