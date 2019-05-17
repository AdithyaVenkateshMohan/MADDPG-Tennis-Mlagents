#!/usr/bin/env python
# coding: utf-8

# # Collaboration and Competition
# 
# ---
# 
# You are welcome to use this coding environment to train your agent for the project.  Follow the instructions below to get started!
# 
# ### 1. Start the Environment
# 
# Run the next code cell to install a few packages.  This line will take a few minutes to run!

# In[1]:


get_ipython().system('pip -q install ./python')


# In[2]:



#from tensorboardX import SummaryWriter


# The environment is already saved in the Workspace and can be accessed at the file path provided below. 

# In[3]:


from unityagents import UnityEnvironment
import numpy as np
path = r"C:\Users\adith\Desktop\Meng Robotics\reinforcement\Banana\MADDPG\Multiagent\Tennis_Windows_x86_64\Tennis"
env = UnityEnvironment(file_name=path)


# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

# In[4]:


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# ### 2. Examine the State and Action Spaces
# 
# Run the code cell below to print some information about the environment.

# In[5]:


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states)


# ### 3. Take Random Actions in the Environment
# 
# In the next code cell, you will learn how to use the Python API to control the agent and receive feedback from the environment.
# 
# Note that **in this coding environment, you will not be able to watch the agents while they are training**, and you should set `train_mode=True` to restart the environment.

# In[6]:


for i in range(5):                                         # play game for 5 episodes
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards
        print("states", actions)
        # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


# In[7]:


import torch
def transpose_list(mylist):
    return list(map(list, zip(*mylist)))

def transpose_to_tensorAsitis(input_list):
    make_tensor = lambda x: torch.tensor(x, dtype=torch.float)
    return list(map(make_tensor, input_list))

env_info = env.step(actions)[brain_name]           # send all actions to tne environment
next_states = env_info.vector_observations
print(next_states)
s = transpose_list(next_states)
print(s)
p = np.concatenate((next_states[0], next_states[1]))
print(np.shape(p)[0])
s = transpose_to_tensorAsitis(next_states)
print("tensor", s)


# In[8]:


# main function that sets up environments
# perform training loop

#import envs
from buffer import ReplayBuffer
from maddpg import MADDPG
import torch
import numpy as np
#from tensorboardX import SummaryWriter
import os
from utilities import transpose_list, transpose_to_tensor

# keep training awake
from workspace_utils import keep_awake

# for saving gif
import imageio

def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)

def pre_process(entity, batchsize):
    processed_entity = []
    for j in range(3):
        list = []
        for i in range(batchsize):
            b = entity[i][j]
            list.append(b)
        c = torch.Tensor(list)
        processed_entity.append(c)
    return processed_entity


def main():
    env_info = env.reset(train_mode=False)[brain_name]
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    
    seeding()
    # number of parallel agents
    #parallel_envs = num_agents
    # number of training episodes.
    # change this to higher number to experiment. say 30000.
    
    number_of_episodes = 10000
    update_actor_after = 100
    update_actor_every = 2
    episode_length = 100
    batchsize = 100
    # how many episodes to save policy and gif
    save_interval = 1000
    t = 0
    
    LR_ACTOR = 1e-5
    LR_CRITIC = 3e-3
    
    # amplitude of OU noise
    # this slowly decreases to 0
    noise = 1.0
    noise_reduction = 0.999999

    # how many episodes before update
    episode_per_update =  1
    no_of_updates_perTime = 1

    log_path = os.getcwd()+"/log"
    model_dir= os.getcwd()+"/model_dir"
    
    os.makedirs(model_dir, exist_ok=True)

    #torch.set_num_threads(parallel_envs)
    #env = envs.make_parallel_env(parallel_envs)
    
    # keep 5000 episodes worth of replay
    buffer = ReplayBuffer(int(10*episode_length))
    
    # initialize policy and critic
    maddpg = MADDPG(lr_actor = LR_ACTOR , lr_critic = LR_CRITIC)
    #logger = SummaryWriter(log_dir=log_path)
    agent0_reward = []
    agent1_reward = []
    #agent2_reward = []

    # training loop
    # show progressbar
    import progressbar as pb
    widget = ['episode: ', pb.Counter(),'/',str(number_of_episodes),' ', 
              pb.Percentage(), ' ', pb.ETA(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ' ]
    
    timer = pb.ProgressBar(widgets=widget, maxval=number_of_episodes).start()

    # use keep_awake to keep workspace from disconnecting
    for episode in range(0, number_of_episodes):

        timer.update(episode)

        env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        reward_this_episode = np.zeros((1, num_agents))
        
        #all_obs = env.reset() #
        obs = states
        obs_full = np.concatenate((states[0], states[1]))

        #for calculating rewards for this particular episode - addition of all time steps

        # save info or not
        save_info = ((episode) % save_interval < 1 or episode==number_of_episodes- 1)
        tmax = 0
        
        #resetting noise 
        for i in range(num_agents):
            maddpg.maddpg_agent[i].noise.reset()
            
        for episode_t in range(episode_length):

            t += 1
            
            update_act = True if (episode > update_actor_after or episode % update_actor_every == 0 ) else False
            # explore = only explore for a certain number of episodes
            # action input needs to be transposed
            actions = maddpg.act(transpose_to_tensorAsitis(obs), noise=noise, batch = False)
            noise *= noise_reduction
            
            actions_array = torch.stack(actions).cpu().detach().numpy()
            
            # transpose the list of list
            # flip the first two indices
            # input to step requires the first index to correspond to number of parallel agents
            actions_for_env = np.rollaxis(actions_array,1)
            
            # step forward one frame
            env_info = env.step(actions_for_env)[brain_name]
            
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            scores += env_info.rewards
            
            rewards_for_env = np.hstack(rewards)
            
            obs = states
            obs_full = np.concatenate((states[0], states[1]))
            next_obs = next_states
            next_obs_full = np.concatenate((next_states[0], next_states[1]))
            # add data to buffer
            transition = (np.array([obs]), np.array([obs_full]), np.array([actions_for_env]), np.array([rewards_for_env]), np.array([next_obs]), np.array([next_obs_full]), np.array([dones] , dtype = 'float'))
            buffer.push(transition)
            
            reward_this_episode += rewards

            obs, obs_full = next_obs, next_obs_full
            
            # update once after every episode_per_update
            if len(buffer) > batchsize and episode % episode_per_update == 0:
                for _ in range(no_of_updates_perTime):
                    for a_i in range(num_agents):
                        samples = buffer.sample(batchsize)
                        #updating the weights of the n/w
                        maddpg.update(samples, a_i , update_actor = update_act)
                    maddpg.update_targets() #soft update the target network towards the actual networks
                
            if np.any(dones):
                # if the episode is done the loop is break to the next episode
                break
        
        for i in range(num_agents):
            agent0_reward.append(reward_this_episode[0,0])
            agent1_reward.append(reward_this_episode[0,1])
            

        if episode % 100 == 0 or episode == number_of_episodes-1:
            avg_rewards = [np.mean(agent0_reward), np.mean(agent1_reward)]
            agent0_reward = []
            agent1_reward = []
            for a_i, avg_rew in enumerate(avg_rewards):
                #logger.add_scalar('agent%i/mean_episode_rewards' % a_i, avg_rew, episode)
                print('agent%i/mean_episode_rewards' % a_i, avg_rew, episode)

        #saving model
        save_dict_list =[]
        if save_info:
            for i in range(num_agents):

                save_dict = {'actor_params' : maddpg.maddpg_agent[i].actor.state_dict(),
                             'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                             'critic_params' : maddpg.maddpg_agent[i].critic.state_dict(),
                             'critic_optim_params' : maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
                save_dict_list.append(save_dict)

                torch.save(save_dict_list, 
                           os.path.join(model_dir, 'episode-{}.pt'.format(episode)))
                
            # save gif files
            #imageio.mimsave(os.path.join(model_dir, 'episode-{}.gif'.format(episode)), 
                            #frames, duration=.04)
    timer.finish()


main()


# When finished, you can close the environment.

# In[9]:


env.close()


# In[10]:


import numpy as np
a = np.array([[1,2,3]])
b = np.array([[0,4,5]])
c = np.concatenate(a+b)
print(c)


# ### 4. It's Your Turn!
# 
# Now it's your turn to train your own agent to solve the environment!  A few **important notes**:
# - When training the environment, set `train_mode=True`, so that the line for resetting the environment looks like the following:
# ```python
# env_info = env.reset(train_mode=True)[brain_name]
# ```
# - To structure your work, you're welcome to work directly in this Jupyter notebook, or you might like to start over with a new file!  You can see the list of files in the workspace by clicking on **_Jupyter_** in the top left corner of the notebook.
# - In this coding environment, you will not be able to watch the agents while they are training.  However, **_after training the agents_**, you can download the saved model weights to watch the agents on your own machine! 

# In[ ]:




