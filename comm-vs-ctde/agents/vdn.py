import torch
from .networks import QNetwork
import random


class VDNAgent:
    def __init__(self, obs_dim, action_dims, hidden_dim, rnn_hidden_dim, lr, gamma, epsilon_start, epsilon_end):
        self.obs_dim = obs_dim
        self.action_dims = action_dims
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end

        self.q_network = QNetwork(obs_dim, action_dims, hidden_dim, rnn_hidden_dim)
        self.target_network = QNetwork(obs_dim, action_dims, hidden_dim, rnn_hidden_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.epsilon = epsilon_start

    def select_action(self, obs, hidden_state, epsilon):
        if random.random() < epsilon:
            actions = [random.randint(0, action_dim - 1) for action_dim in self.action_dims]
            return actions, hidden_state
        else:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0) # (obs_dim,) --> (batch_size, seq_len, obs_dim)
                # hidden_state is dim (num_layers, rnn_hidden_state_dim) --> (num_layers, batch_size, rnn_hidden_state_dim)
                hidden_state_tensor = torch.FloatTensor(hidden_state).unsqueeze(1) if hidden_state is not None else self.q_network.init_hidden(1)
                q_values, new_hidden_state = self.q_network(obs_tensor, hidden_state_tensor)
                actions = [torch.argmax(q_vals, dim=-1).item() for q_vals in q_values]
                return actions, new_hidden_state.squeeze(1).numpy() # back to (num_layers, rnn_hidden_state_dim)

    # TODO: Potentially handle entire batch at once if training is slow
    def train(self, batch):
        """
        batch: list of episodes from replay buffer
        Each episode: {'obs', 'actions', 'rewards', 'next_obs', 'dones', 'hidden_states'}
        """
        total_loss = 0
        # For each episode in batch:
        for episode in batch:
            obs = torch.FloatTensor(episode['obs'])  # (ep_len, num_agents, obs_dim)
            actions = torch.LongTensor(episode['actions'])
            rewards = torch.FloatTensor(episode['rewards']).unsqueeze(-1)
            next_obs = torch.FloatTensor(episode['next_obs'])
            dones = torch.FloatTensor(episode['dones']).unsqueeze(-1)
            hidden_states = torch.FloatTensor(episode['hidden_states'])

            # Stack agents into batch dimension: (ep_len, num_agents, obs_dim) -> (ep_len * num_agents, obs_dim)
            ep_len, num_agents, obs_dim = obs.shape
            obs_stacked = obs.reshape(ep_len * num_agents, obs_dim).unsqueeze(1)  # (ep_len*num_agents, 1, obs_dim)
            next_obs_stacked = next_obs.reshape(ep_len * num_agents, obs_dim).unsqueeze(1)

            # Expand hidden states to match stacked batch size
            # (num_agents, num_layers, hidden_dim) -> (num_layers, num_agents, hidden_dim) -> (num_layers, ep_len*num_agents, hidden_dim)
            init_hidden_per_agent = hidden_states[0].permute(1, 0, 2)  # (1, num_agents, 64)
            init_hidden = init_hidden_per_agent.repeat(1, ep_len, 1)  # (1, ep_len*num_agents, 64)

            # Forward pass for all agents at once
            q_values, _ = self.q_network(obs_stacked, init_hidden)
            # q_values[0]: (ep_len*num_agents, 1, 5), q_values[1]: (ep_len*num_agents, 1, 10)

            # Reshape back: (ep_len*num_agents, 1, action_dim) -> (ep_len, num_agents, action_dim)
            movement_q_values = q_values[0].squeeze(1).reshape(ep_len, num_agents, -1)
            comm_q_values = q_values[1].squeeze(1).reshape(ep_len, num_agents, -1)
            # (ep_len, num_agents, 1)
            movement_actions = (actions[:, :, 0]).unsqueeze(-1)
            # (ep_len, num_agents, 1)
            comm_actions = (actions[:, :, 1]).unsqueeze(-1)
            # (ep_len, num_agents, 1)
            movement_q_values = torch.gather(movement_q_values, dim=2, index=movement_actions)
            # Sum movement Q values across agent dim
            movement_q_total = movement_q_values.sum(dim=1, keepdim=True)
            # (ep_len, num_agents, 1)
            comm_q_values = torch.gather(comm_q_values, dim=2, index=comm_actions)
            # Sum comm Q values across agent dim
            comm_q_total = comm_q_values.sum(dim=1, keepdim=True)
            # compute target q-values
            with torch.no_grad():
                # Double Q-Learning: use Q-network to select actions, target network to evaluate
                next_q_values, _ = self.q_network(next_obs_stacked, init_hidden)
                target_q_values, _ = self.target_network(next_obs_stacked, init_hidden)

                # Reshape: (ep_len*num_agents, 1, action_dim) -> (ep_len, num_agents, action_dim)
                movement_next_q_values = next_q_values[0].squeeze(1).reshape(ep_len, num_agents, -1)
                comm_next_q_values = next_q_values[1].squeeze(1).reshape(ep_len, num_agents, -1)
                movement_target_q_values = target_q_values[0].squeeze(1).reshape(ep_len, num_agents, -1)
                comm_target_q_values = target_q_values[1].squeeze(1).reshape(ep_len, num_agents, -1)

                # Select best actions from Q-network
                best_movement_actions = torch.argmax(movement_next_q_values, dim=2).unsqueeze(-1)
                best_comm_actions = torch.argmax(comm_next_q_values, dim=2).unsqueeze(-1)
                # (ep_len, num_agents, 1)
                target_movement_q_values = torch.gather(movement_target_q_values, dim=2, index=best_movement_actions)
                target_movement_q_values = target_movement_q_values * (1 - dones)
                # Sum target movement Q values across agent dim
                target_movement_q_total = target_movement_q_values.sum(dim=1, keepdim=True)
                # (ep_len, num_agents, 1)
                target_comm_q_values = torch.gather(comm_target_q_values, dim=2, index=best_comm_actions)
                target_comm_q_values = target_comm_q_values * (1 - dones)
                # Sum target comm Q values across agent dim
                target_comm_q_total = target_comm_q_values.sum(dim=1, keepdim=True)
            # compute team rewards (sum across agent dim)
            team_rewards = rewards.sum(dim=1, keepdim=True)
            # compute TD-error/loss (MSE)
            episode_loss = torch.nn.functional.mse_loss(movement_q_total, team_rewards + self.gamma * target_movement_q_total) + \
                torch.nn.functional.mse_loss(comm_q_total, team_rewards + self.gamma * target_comm_q_total)
            total_loss += episode_loss

        # backprop and update Q-network
        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Compute mean Q-values for logging (from last episode in batch)
        mean_q_tot = (movement_q_total.mean() + comm_q_total.mean()).item() / 2

        return total_loss.item() / len(batch), grad_norm.item(), mean_q_tot

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
