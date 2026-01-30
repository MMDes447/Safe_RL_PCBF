import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Beta, Categorical
from torch.distributions import Beta, Categorical
import torch.nn.functional as F
class ActorNetwork1(nn.Module):
    def __init__(self, n_actions, continuous_dim=1, input_dims=3, alpha=0.0003, 
                 fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(ActorNetwork1, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_ppo')
        
        self.shared_network = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU()
        )
        
        self.categorical_head = nn.Sequential(
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )
        
        # Beta distribution parameters
        self.alpha_head = nn.Linear(fc2_dims, continuous_dim)
        self.beta_head = nn.Linear(fc2_dims, continuous_dim)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        shared_features = self.shared_network(state)
        
        action_probs = self.categorical_head(shared_features)
        categorical_dist = Categorical(action_probs)
        
        # Beta(alpha, beta) with alpha, beta > 1 for unimodal
        alpha = F.softplus(self.alpha_head(shared_features)) + 1.0
        beta = F.softplus(self.beta_head(shared_features)) + 1.0
        continuous_dist = Beta(alpha, beta)
        
        return categorical_dist, continuous_dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))