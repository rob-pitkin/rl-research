import torch
from .networks import QNetwork
from utils.running_mean_std import RunningMeanStd
import random


class IQLAgent:
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
        batch: list of episodes from replay buffer
        Each episode: {'obs', 'actions', 'rewards', 'next_obs', 'terminated', 'truncated', 'hidden_states'}

        Recurrent unroll: each agent's full episode is one GRU sequence
        (batch_size_seq = num_agents, seq_len = ep_len). Hidden state flows
        across timesteps so the GRU actually integrates history.
        """
        total_loss = 0
        last_mov_q = last_com_q = None
        for episode in batch:
            obs = torch.FloatTensor(episode['obs'])           # (T, N, obs_dim)
            actions = torch.LongTensor(episode['actions'])    # (T, N, 2)
            rewards = torch.FloatTensor(episode['rewards'])   # (T, N)
            if self.standardise_rewards:
                self.rew_ms.update(rewards)
                rewards = (rewards - self.rew_ms.mean) / torch.sqrt(self.rew_ms.var + 1e-8)
            next_obs = torch.FloatTensor(episode['next_obs']) # (T, N, obs_dim)
            terminated = torch.FloatTensor(episode['terminated'])  # (T, N)
            # NOTE: truncated is intentionally unused in the Bellman mask.
            # Time-limit truncation is NOT a real terminal — we should still
            # bootstrap Q(s_T, ·) rather than treat it as 0.

            T, N, obs_dim = obs.shape

            # GRU expects (batch=N, seq=T, obs_dim). Permute time-then-agent
            # data into agent-then-time so each agent is one sequence.
            obs_seq = obs.permute(1, 0, 2)              # (N, T, obs_dim)
            next_obs_seq = next_obs.permute(1, 0, 2)    # (N, T, obs_dim)
            # Fresh zero hidden state at the start of every replayed episode.
            h0 = self.q_network.init_hidden(N)          # (1, N, H)

            # Forward pass: online net on obs sequence — get Q(s_t, a)
            online_movement_seq, online_comm_seq = self._forward_seq(self.q_network, obs_seq, h0)
            # Shapes: (N, T, 5), (N, T, 10). Permute back to (T, N, ·).
            online_mov_q = online_movement_seq.permute(1, 0, 2)
            online_com_q = online_comm_seq.permute(1, 0, 2)

            # Q for actions actually taken
            mov_a = actions[:, :, 0].unsqueeze(-1)     # (T, N, 1)
            com_a = actions[:, :, 1].unsqueeze(-1)
            chosen_mov_q = online_mov_q.gather(2, mov_a)   # (T, N, 1)
            chosen_com_q = online_com_q.gather(2, com_a)

            # Targets: forward both networks on next_obs sequence (no grad)
            with torch.no_grad():
                online_mov_next, online_com_next = self._forward_seq(self.q_network, next_obs_seq, h0)
                target_mov_next, target_com_next = self._forward_seq(self.target_network, next_obs_seq, h0)

                # Permute back to (T, N, ·)
                online_mov_next = online_mov_next.permute(1, 0, 2)
                online_com_next = online_com_next.permute(1, 0, 2)
                target_mov_next = target_mov_next.permute(1, 0, 2)
                target_com_next = target_com_next.permute(1, 0, 2)

                # Double-Q: argmax via online, evaluate via target
                best_mov = online_mov_next.argmax(dim=2, keepdim=True)
                best_com = online_com_next.argmax(dim=2, keepdim=True)
                tgt_mov_q = target_mov_next.gather(2, best_mov)   # (T, N, 1)
                tgt_com_q = target_com_next.gather(2, best_com)

                # Mask ONLY on true termination (not truncation)
                not_terminal = (1.0 - terminated).unsqueeze(-1)   # (T, N, 1)
                tgt_mov_q = tgt_mov_q * not_terminal
                tgt_com_q = tgt_com_q * not_terminal

                rewards_t = rewards.unsqueeze(-1)                 # (T, N, 1)
                target_mov_full = rewards_t + self.gamma * tgt_mov_q
                target_com_full = rewards_t + self.gamma * tgt_com_q

            ep_loss = torch.nn.functional.mse_loss(chosen_mov_q, target_mov_full) + \
                      torch.nn.functional.mse_loss(chosen_com_q, target_com_full)
            total_loss += ep_loss
            last_mov_q, last_com_q = chosen_mov_q, chosen_com_q

        # Mean across episodes, not sum — otherwise effective lr scales with batch_size.
        loss = total_loss / len(batch)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        mean_q = (last_mov_q.mean() + last_com_q.mean()).item() / 2
        return loss.item(), grad_norm.item(), mean_q

    @staticmethod
    def _forward_seq(net, obs_seq, h0):
        """Run the QNetwork over a full sequence. obs_seq: (N, T, obs_dim).

        Returns per-head Q at every timestep, shape (N, T, action_dim).
        This relies on the GRU processing the full sequence at once and
        applying the Q-heads to every step (not just the last).
        """
        # GRU forward over the whole sequence
        gru_out, _ = net.rnn(obs_seq, h0)               # (N, T, H)
        # Apply each Q-head at every timestep
        q_heads = [head(gru_out) for head in net.q_heads]  # each (N, T, A_i)
        return q_heads[0], q_heads[1]

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
