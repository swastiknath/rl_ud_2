# ***Continuous Control*** 
***Udacity Deep Reinforcement Learning***

***Capstone 2: Continuous Control***

***Swastik Nath***


---

### Words About the Environment:

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

We have two separate versions of the Unity ML-Agent environment:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  


### Solving the Environment


#### Solving the Version 1 : Single-Agent Environment: 

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

#### Solving the Version 2: Multi-Agent Environment:

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 


### Algorithmic Approach:

To solve the environment we use an Actor-Critic architecture with DDPG Algorithm. We use a total of 4 Deep Neural Networks:

***Actor Networks***: We use a total of 2 Actor Networks which works by estimating Policy Gradients.   

 - Actor Network (Online) : Interacts with the Environment Real-time. It uses two Linear Layers with Batch Normalization in between them. It uses 128 neurons in the hidden layers. Between the linear layers we use the ***Relu*** activation. We use ***Tanh*** as the final activation layer. We feed the state from the environment to the Actor online model. 
 ```
 Actor(
  (fc1): Linear(in_features=33, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=4, bias=True)
  (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
 ```

 - Actor Network (Target): The online model weights are copied to this model at a few timesteps. It uses two Linear Layers with Batch Normalization in between them. It uses 128 neurons in the hidden layers. Between the linear layers we use the ***Relu*** activation. The structure of the target network is totally similar to that of the online network.


***Critic Network***: We use 2 another Deep Neural Networks to estimate and evaluate the Action values for state, action pair. 

 - Critic Network (Online): The weights from the Actor Online Model is copied here after a few specified timesteps. We use a 128 neurons between the first layer and second layer, after which we add the size of actions and the number of hidden layers to the second linear layer. We use the output of the last linear layer as the output of the network.  
 ```
 Critic(
  (fc1): Linear(in_features=33, out_features=128, bias=True)
  (fc2): Linear(in_features=132, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=1, bias=True)
  (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
 ```
 - Critic Network (Target):  The weights from the Actor Target Model is copied here after a few specified timesteps. We use a 128 neurons between the first layer and second layer, after which we add the size of actions and the number of hidden layers to the second linear layer. We use the output of the last linear layer as the output of the network. The structure of the target network is totally similar to that of the online network.

 
It is not possible to straightforwardly apply Q-learning to continuous action spaces, because in continuous spaces finding the greedy policy requires an optimization of the action at every timestep; this optimization is too slow to be practical with large, unconstrained function approximators and nontrivial
action spaces. Instead, here we used an actor-critic approach based on the DPG algorithm. 

The DPG algorithm maintains a parameterized actor function µ(s|θ
µ) which specifies the current
policy by deterministically mapping states to a specific action. The critic Q(s, a) is learned using
the Bellman equation as in Q-learning.

A major challenge of learning in continuous action spaces is exploration. An advantage of offpolicies algorithms such as DDPG is that we can treat the problem of exploration independently
from the learning algorithm. We constructed an exploration policy µ
0 by adding noise sampled from a noise process N to our actor policy
µ0(st) = µ(st|θ;µ;t) + N (7)

N can be chosen to suit the environment. We use the ***Ornstein-Uhlenbeck process*** (Uhlenbeck & Ornstein, 1930) to generate the noises and induce them to the states for further exploratory analysis and randomness. 


### Results:

 - #### Version 1: Single Agent Version: 

    The Single Agent version of the Implementation was able to solve the environment in just **114** episodes with Average Score of 30.04 when the training ended. It took around 45 minutes to train on GPU enabled environment. 

    ![single_image](https://github.com/swastiknath/rl_ud_2/raw/master/single_agent.png)


 - ### Version 2: Multi Agent Version: 
   
   The Multi-Agent Version of the Implementation trains multiple agents to interact with the environment. It gave average of the scores as below:
   
   | Number of Episodes   | Average Score |
   |-----------------------|--------------|
   | 200                   | 37.86 |
   |300 | 38.25 |
   |400   | 38.09 |
   |452   | 38.20   |
  
   To train 200 epochs it took around 1 hour and 17 minutes on GPU enabled device. 

   ![multiple_agent](https://github.com/swastiknath/rl_ud_2/raw/master/ddpg_multiagent.png)

### Hyperparameters:
We use the following set of hyperparameters with the multi and single agent environment. 

| Hyperparameter | Single Agent Environment | Multi Agent Environment |
|----------------|--------------------------|-------------------------|
| Buffer Memory Size |   1e-5     |    1e-5       |
| Batch Size for Experience Replay |       128           |     128          |
| Discount Factor (Gamma)             |     0.99         |       0.99       |
| Interpolation Factor (TAU)          |     1e-3         |     1e-3       |
| Learning Rate for Actor (LR) Adam Optimizer      |   2e-4         |   2e-4     |
| Learning Rate for Critic (LR) Adam Optimizer    |       2e-4       |  2e-4    |
| Weight Decay                |      0             |    0             |
| MU (Mean for Orstein Uhlenbeck Noise)   |   0      |   0       |
|Sigma (Standard Deviation for Orstein Uhlenbeck Noise)|    0.1     |   0.1            |
|Theta for Orstein Uhlenbeck Noise) |     0.15      |    0.15      |
| 1st Actor Linear Layer Hidden Size | 128          | 128 |
| 1st Critic Linear Layer Hidden Size | 128 | 128 |
| 2nd Actor Linear Layer Hidden Size | 128   | 128 |
| 2nd Critic Linear Layer Hidden Size | 128+4(Actions) = 132 | 128 +4(Actions) = 132 |

### Future Ideas :

The performance and inference speed of the archietecture can get better with the following few future Ideas:
 - Change in Algorithm : Instead of Using the Actor-Critic DDPG we can use D4PG, PPO with further enhancements for getting better performance. 
 - Change in Hyperparameters: To conform with the new algorithm we might need to tweak the hyperparameters futher to get better efficiency. 

