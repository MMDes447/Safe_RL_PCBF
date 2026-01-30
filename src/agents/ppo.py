import numpy as np
import torch as T
import torch.nn.functional as F

from networks.actor import ActorNetwork1
from networks.critic import CriticNetwork1
from networks.memory import PPOMemory1

class PPOAgent1_epch:
    def __init__(self, n_actions, input_dims, continuous_dim=1, alpha=0.0003, batch_size=32,
                n_epochs=30, gae_lambda=0.95, gamma=0.99, policy_clip=0.2):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.global_step = 0
        self.epsilon = .01
        self.continuous_dim = continuous_dim  # Store the continuous action dimension

        # PPO networks - updated to support continuous actions
        self.actor = ActorNetwork1(n_actions, continuous_dim, input_dims, alpha)  # Modified actor
        self.critic = CriticNetwork1(input_dims, alpha)
        self.memory = PPOMemory1(batch_size)  # Need a new memory class for hybrid actions

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        
        discrete_dist, continuous_dist = self.actor(state)
        value = self.critic(state)
        
        discrete_action = discrete_dist.sample()
        continuous_action = continuous_dist.sample()  # Already in [0, 1]!
        
        discrete_log_prob = discrete_dist.log_prob(discrete_action)
        continuous_log_prob = continuous_dist.log_prob(continuous_action).sum(dim=-1)
        total_log_prob = discrete_log_prob + continuous_log_prob
        
        discrete_action_val = T.squeeze(discrete_action).item()
        continuous_action_val = T.squeeze(continuous_action).cpu().detach().numpy()
        total_log_prob_val = T.squeeze(total_log_prob).item()
        value_val = T.squeeze(value).item()
        
        return discrete_action_val, continuous_action_val, total_log_prob_val, value_val

    def remember(self, state, discrete_action, continuous_action,Mpc_continous_actions,Mpc_discrete_actions, log_prob, val, reward, done, next_state):
        # Store in hybrid memory
        self.memory.store_memory(state, discrete_action, continuous_action,Mpc_continous_actions,Mpc_discrete_actions, log_prob, val, reward, done)
        
        

    def learn(self):
        for _ in range(self.n_epochs):
            # Generate batches with both action types
            state_arr, discrete_action_arr, continuous_action_arr,Mpc_continous_actions_arr,Mpc_discrete_actions_arr, old_prob_arr, vals_arr, reward_arr, done_arr, batches = \
                self.memory.generate_batches()

            # GAE calculation remains the same
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*
                    (1-int(done_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t

            advantage = T.tensor(advantage).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)

            # PPO update with both action types
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                discrete_actions = T.tensor(discrete_action_arr[batch], dtype=T.int64).to(self.actor.device)
                continuous_actions = T.tensor(continuous_action_arr[batch], dtype=T.float).to(self.actor.device)
                Mpc_continous_actions= T.tensor(Mpc_continous_actions_arr[batch], dtype=T.float).to(self.actor.device)
                Mpc_discrete_actions= T.tensor(Mpc_discrete_actions_arr[batch], dtype=T.int64).to(self.actor.device)

                
                # Get distributions
                discrete_dist, continuous_dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                # Get log probabilities for both action types
                discrete_log_probs = discrete_dist.log_prob(discrete_actions)
                continuous_log_probs = continuous_dist.log_prob(continuous_actions).sum(dim=-1)
                dist_entropy=discrete_dist.entropy().mean()+continuous_dist.entropy().mean()
                # entropy_coef=0.01
                # Total new log probs
                new_probs = discrete_log_probs + continuous_log_probs

                # Calculate importance sampling ratio
                prob_ratio = (new_probs - old_probs).exp()

                # PPO actor loss components - same as before
                weighted_probs = advantage[batch]*prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                1+self.policy_clip)*advantage[batch]
                light_imitation_loss = F.mse_loss(continuous_actions, Mpc_continous_actions)
                blind_imitation_loss = F.cross_entropy(discrete_dist.logits, Mpc_discrete_actions)

                # Original PPO actor loss
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                constrained_actor_loss = actor_loss
                
                # Critic loss - same as before
                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()
                loss= constrained_actor_loss + 0.5*critic_loss - 0.01*dist_entropy+light_imitation_loss+blind_imitation_loss

                # Update networks
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                loss.backward()
                # constrained_actor_loss.backward()
                # critic_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                
                self.global_step+=1
                # print(f"Actor loss: {actor_loss.item()}")

        # Clear memory at the end
        self.memory.clear_memory()