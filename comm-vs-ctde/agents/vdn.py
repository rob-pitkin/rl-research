import torch
from .networks import QNetwork
from utils.running_mean_std import RunningMeanStd
import random


class VDNAgent:
    def __init__(self, obs_dim, action_dims, hidden_dim, rnn_hidden_dim, lr, gamma, epsilon_start, epsilon_end,
                 num_agents=2, standardise_rewards=True):
        self.obs_dim = obs_dim
        self.action_dims = action_dims
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.num_agents = num_agents
        self.standardise_rewards = standardise_rewards

        self.q_network = QNetwork(obs_dim, action_dims, hidden_dim, rnn_hidden_dim)
        self.target_network = QNetwork(obs_dim, action_dims, hidden_dim, rnn_hidden_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.epsilon = epsilon_start
        if self.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(self.num_agents,))

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

    def train(self, batch):
        """
        VDN with recurrent unroll. Each agent's full episode is one GRU
        sequence. After computing per-agent Q, sum across agents (the VDN
        mixer) to get Q_tot, fit against the team-summed reward.
        """
        total_loss = 0
        last_mov_q_tot = last_com_q_tot = None
        for episode in batch:
            obs = torch.FloatTensor(episode['obs'])
            actions = torch.LongTensor(episode['actions'])
            rewards = torch.FloatTensor(episode['rewards'])  # (T, N)
            if self.standardise_rewards:
                self.rew_ms.update(rewards)
                rewards = (rewards - self.rew_ms.mean) / torch.sqrt(self.rew_ms.var + 1e-8)
            next_obs = torch.FloatTensor(episode['next_obs'])
            terminated = torch.FloatTensor(episode['terminated'])  # (T, N)

            T, N, obs_dim = obs.shape
            obs_seq = obs.permute(1, 0, 2)            # (N, T, obs_dim)
            next_obs_seq = next_obs.permute(1, 0, 2)
            h0 = self.q_network.init_hidden(N)

            online_mov_seq, online_com_seq = self._forward_seq(self.q_network, obs_seq, h0)
            online_mov_q = online_mov_seq.permute(1, 0, 2)   # (T, N, 5)
            online_com_q = online_com_seq.permute(1, 0, 2)

            mov_a = actions[:, :, 0].unsqueeze(-1)
            com_a = actions[:, :, 1].unsqueeze(-1)
            chosen_mov_q = online_mov_q.gather(2, mov_a)     # (T, N, 1)
            chosen_com_q = online_com_q.gather(2, com_a)

            # VDN mix: sum across agent dim → (T, 1, 1)
            mov_q_tot = chosen_mov_q.sum(dim=1, keepdim=True)
            com_q_tot = chosen_com_q.sum(dim=1, keepdim=True)

            with torch.no_grad():
                online_mov_next, online_com_next = self._forward_seq(self.q_network, next_obs_seq, h0)
                target_mov_next, target_com_next = self._forward_seq(self.target_network, next_obs_seq, h0)
                online_mov_next = online_mov_next.permute(1, 0, 2)
                online_com_next = online_com_next.permute(1, 0, 2)
                target_mov_next = target_mov_next.permute(1, 0, 2)
                target_com_next = target_com_next.permute(1, 0, 2)

                best_mov = online_mov_next.argmax(dim=2, keepdim=True)
                best_com = online_com_next.argmax(dim=2, keepdim=True)
                tgt_mov_q = target_mov_next.gather(2, best_mov)   # (T, N, 1)
                tgt_com_q = target_com_next.gather(2, best_com)

                not_terminal = (1.0 - terminated).unsqueeze(-1)   # (T, N, 1)
                tgt_mov_q = tgt_mov_q * not_terminal
                tgt_com_q = tgt_com_q * not_terminal

                # VDN mix on the target side too
                tgt_mov_q_tot = tgt_mov_q.sum(dim=1, keepdim=True)
                tgt_com_q_tot = tgt_com_q.sum(dim=1, keepdim=True)

                team_rewards = rewards.sum(dim=1, keepdim=True).unsqueeze(-1)  # (T, 1, 1)
                target_mov_full = team_rewards + self.gamma * tgt_mov_q_tot
                target_com_full = team_rewards + self.gamma * tgt_com_q_tot

            ep_loss = torch.nn.functional.mse_loss(mov_q_tot, target_mov_full) + \
                      torch.nn.functional.mse_loss(com_q_tot, target_com_full)
            total_loss += ep_loss
            last_mov_q_tot, last_com_q_tot = mov_q_tot, com_q_tot

        loss = total_loss / len(batch)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        mean_q_tot = (last_mov_q_tot.mean() + last_com_q_tot.mean()).item() / 2
        return loss.item(), grad_norm.item(), mean_q_tot

    @staticmethod
    def _forward_seq(net, obs_seq, h0):
        gru_out, _ = net.rnn(obs_seq, h0)
        q_heads = [head(gru_out) for head in net.q_heads]
        return q_heads[0], q_heads[1]

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
