from collections import deque
import random


class EpisodeReplayBuffer:
    def __init__(self, max_capacity):
        self.buffer = deque()
        self.max_capacity = max_capacity

    def add_episode(self, obs, actions, rewards, next_obs, dones, hidden_states):
        episode = {
            "obs": obs, # (episode_len, num_agents, obs_dim)
            "actions": actions, # (episode_len, num_agents, action_dim)
            "rewards": rewards, # (episode_len, num_agents, 1)
            "next_obs": next_obs, # (episode_len, num_agents, obs_dim)
            "dones": dones, # (episode_len, num_agents, 1)
            "hidden_states": hidden_states, # (episode_len, num_agents, rnn_hidden_dim)
        }
        self.buffer.appendleft(episode)
        if len(self.buffer) > self.max_capacity:
            self.buffer.pop()


    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        indices = random.sample(range(len(self.buffer)), batch_size)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)
