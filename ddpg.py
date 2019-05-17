# individual network settings for each actor + critic pair
# see networkforall for details

from networkforall import Network
from utilities import hard_update, gumbel_softmax, onehot_from_logits
from torch.optim import Adam
import torch
import numpy as np


# add OU noise for exploration
from OUNoise import OUNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#device = 'cpu'

class DDPGAgent:
    def __init__(self, state_size , action_size, hidden_in_dim, hidden_out_dim, num_agents =2 , lr_actor=1.0e-4, lr_critic=1.0e-3):
        super(DDPGAgent, self).__init__()
        critic_state_size = state_size*num_agents + (action_size *(num_agents -1))
        self.actor = Network(state_size , action_size, hidden_in_dim, hidden_out_dim , actor=True).to(device)
        self.critic = Network(critic_state_size , action_size , hidden_in_dim, hidden_out_dim).to(device)
        self.target_actor = Network(state_size  , action_size, hidden_in_dim, hidden_out_dim, actor=True).to(device)
        self.target_critic = Network(critic_state_size, action_size , hidden_in_dim, hidden_out_dim ).to(device)

        self.noise = OUNoise(action_size, scale=1.0 )

        
        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=0)
        
        print("critic", self.critic , self.target_critic ,"optim" ,self.critic_optimizer)
        print("actor", self.actor , self.target_actor , "optim", self.actor_optimizer)


    def act(self, obs, noise=0.0 , batch = True):
        obs = obs.to(device)
        #self.actor.eval()
        act = self.actor(obs , batch = batch).cpu().data
        no = noise*self.noise.noise()
        print( "act" , act , "noise" , no)
        action = act + no
        return np.clip(action,-1,1)

    def target_act(self, obs, noise=0.0 , batch = True):
        obs = obs.to(device)
        #self.target_actor.eval()
        action = self.target_actor(obs , batch = batch).cpu().data + noise*self.noise.noise()
        return np.clip(action,-1,1)
