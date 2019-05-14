# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
from utilities import soft_update, transpose_to_tensor, transpose_list , transpose_to_tensorAsitis
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
import numpy as np



class MADDPG:
    def __init__(self, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 48+2+2=52
        # have to change the agent neurons for sure
        
        # the no of agents is two because there are only two players
        self.maddpg_agent = [DDPGAgent(24, 512, 256, 2, 52, 512, 256), 
                             DDPGAgent(24, 512, 256, 2, 52, 512, 256)]
        
        self.num_agents = 2 
        self.action_vector = 2
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def get_actors(self):
        """get target_actors of all """
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        #print(obs_all_agents)
        #shape_vec = [np.shape(obs) for obs in obs_all_agents]
        #print("shape",shape_vec)
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self,agent_no, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target = []
        for i, obs in enumerate(obs_all_agents):
                target_actions = [self.maddpg_agent[i].target_act(obs[i,:]) for i in range(agent_no)]
                target_actions = torch.stack(target_actions)
                target.append(target_actions)
        return target
    
    def target_act_batch(self,agent_no, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [self.maddpg_agent[i].target_act(obs_all_agents[:,i,:]) for i in range(agent_no)]
        target_actions = torch.stack(target_actions)
        return target_actions
    
    def act_with_agent(self,agent_no, agent_id ,obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        
        actions = [self.maddpg_agent[i].actor(obs_all_agents[:,i,:]) if i == agent_id \
           else self.maddpg_agent[i].actor(obs_all_agents[:,i,:]).detach() for i in range(agent_no)]
        
        actions = torch.stack(actions)

        return actions

    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        
        
        
        obs, obs_full, action, reward, next_obs, next_obs_full, done = samples
        samples = (np.array(obs), obs_full, action, reward, np.array(next_obs), next_obs_full, done)
        
        batch_size = np.shape(obs_full)[0]
        
        action_size = self.num_agents * self.action_vector
        
        obs, obs_full, action, reward, next_obs, next_obs_full, done = map(transpose_to_tensorAsitis, samples)

        obs_full = torch.stack(obs_full).to(device)
        next_obs_full = torch.stack(next_obs_full).to(device)
        
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        # critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        # !crictal logic error have to change her to the agents observation only
        
        
        next_obs = torch.stack(next_obs).to(device)
        obs = torch.stack(obs).to(device)
        
        
        target_actions = self.target_act_batch(2,next_obs)
        
        target_actions = target_actions.to(device)
        
        #batch size and action size
        target_critic_input = torch.cat((next_obs_full.view(-1,batch_size).t(),target_actions.view(-1,action_size)), dim=1).to(device)
        
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input).to(device)
        
        indices = torch.tensor([1])
        reward = torch.stack(reward).to(device)
        done = torch.stack(done).to(device)
        
        y = reward[:,agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[:,agent_number].view(-1, 1)).to(device)
        
        action = torch.stack(action).to(device)
        
        critic_input = torch.cat((obs_full.view(-1,batch_size).t(), action.view(-1,action_size)), dim=1).to(device)
        q = agent.critic(critic_input).to(device)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1)
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        
        
#         q_input = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
#                    else self.maddpg_agent[i].actor(ob).detach()
#                    for i, ob in enumerate(obs) ]

        q_input = self.act_with_agent(2,agent_number,obs)
                
        q_input = q_input.to(device)
        
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        
        q_input2 = torch.cat((obs_full.view(-1,batch_size).t(), q_input.view(-1,action_size)), dim=1).to(device)
        
        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),1)
        agent.actor_optimizer.step()

#         al = actor_loss.cpu().detach().item()
#         cl = critic_loss.cpu().detach().item()
#         logger.add_scalars('agent%i/losses' % agent_number,
#                            {'critic loss': cl,
#                             'actor_loss': al},
#                            self.iter)

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
            
            
            




