import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dims, hidden_dim, rnn_hidden_dim):
        super(QNetwork, self).__init__()
        self.obs_dim = obs_dim
        self.action_dims = action_dims
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        # Input size: (batch_size, seq_len, obs_dim)
        self.rnn = nn.GRU(self.obs_dim, self.rnn_hidden_dim, batch_first=True)
        # Output size: (batch_size, action_dim)
        self.q_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.rnn_hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, action_dim)
            )
            for action_dim in self.action_dims
        ])

    def forward(self, obs, hidden_state):
        """
        Returns Q-values and the new hidden state
        """
        output, enc = self.rnn(obs, hidden_state)
        # Output is size (batch_size, seq_len, rnn_hidden_dim)
        # Since seq_len is typically 1, select only the last timestep
        output = output[:, -1, :]
        q_values = [q_head(output) for q_head in self.q_heads]
        # q_values is list of: [Q_movement (batch, 5), Q_comm (batch, 10)]
        return q_values, enc

    def init_hidden(self, batch_size, device='cpu'):
      """
      Returns an initialized hidden state for the GRU.
      """
      return torch.zeros(1, batch_size, self.rnn_hidden_dim, device=device)




