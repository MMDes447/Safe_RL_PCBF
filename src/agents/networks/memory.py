class PPOMemory1:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.discrete_actions = []
        self.continuous_actions = []  # New list for continuous actions
        self.Mpc_continous_actions = []  # New list for MPC continuous actions
        self.Mpc_discrete_actions = []  # New list for MPC discrete actions
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
        
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return np.array(self.states), np.array(self.discrete_actions), np.array(self.continuous_actions),np.array(self.Mpc_continous_actions),np.array(self.Mpc_discrete_actions), \
               np.array(self.probs), np.array(self.vals), np.array(self.rewards), np.array(self.dones), \
               batches
    
    def store_memory(self, state, discrete_action, continuous_action,Mpc_continous_actions,Mpc_discrete_actions, probs, vals, reward, done):
        self.states.append(state)
        self.discrete_actions.append(discrete_action)
        self.continuous_actions.append(continuous_action)
        self.Mpc_continous_actions.append(Mpc_continous_actions)  # New list for MPC continuous actions
        self.Mpc_discrete_actions.append(Mpc_discrete_actions)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def clear_memory(self):
        self.states = []
        self.probs = []
        self.discrete_actions = []
        
        self.continuous_actions = []
        self.Mpc_continous_actions = []  # New list for MPC continuous actions
        self.Mpc_discrete_actions = [] 
        self.rewards = []
        self.dones = []
        self.vals = []