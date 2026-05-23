import numpy as np
from cpp_mpe2.simple_reference import simple_reference


class SimpleReferenceWrapper:
    def __init__(self, mask_communication=False, obs_agent_id=True):
        self.env = simple_reference.parallel_env()
        self.mask_communication = mask_communication
        self.obs_agent_id = obs_agent_id
        self.agents = None
        self.num_agents = 2
        self.dim_c = 10

    def reset(self):
        obs_dict, info_dict = self.env.reset()
        self.agents = self.env.agents
        return [
            self._append_agent_id(self._mask_observation(obs_dict[agent]), i)
            for i, agent in enumerate(self.agents)
        ]

    def step(self, actions):
        # Convert factored actions [movement, comm] to flat actions
        masked_actions = [self._mask_action(action) for action in actions]
        flat_actions = [self._factored_to_flat(action) for action in masked_actions]
        action_dict = {agent: flat_actions[i] for i, agent in enumerate(self.agents)}
        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = self.env.step(action_dict)
        obs_list = [
            self._append_agent_id(self._mask_observation(obs_dict[agent]), i)
            for i, agent in enumerate(self.agents)
        ]
        reward_list = [reward_dict[agent] for agent in self.agents]
        terminated_list = [terminated_dict[agent] for agent in self.agents]
        truncated_list = [truncated_dict[agent] for agent in self.agents]
        info_list = [info_dict[agent] for agent in self.agents]
        return obs_list, reward_list, terminated_list, truncated_list, info_list

    def get_action_dim(self):
        # Returns [movement_dim, comm_dim] for factored action space
        # Simple reference: [say_0...say_9] X [no_action, left, right, down, up]
        # Cartesian product gives Discrete(50)
        return [5, 10]  # [movement_dim, comm_dim]

    def _factored_to_flat(self, factored_action):
        """Convert [movement, comm] to single flat action (comm * 5 + movement)"""
        movement, comm = factored_action
        return comm * 5 + movement

    def _flat_to_factored(self, flat_action):
        """Convert single flat action to [movement, comm]"""
        movement = flat_action % 5
        comm = flat_action // 5
        return [movement, comm]

    def get_obs_dim(self):
        if self.agents is None:
            self.env.reset()
            self.agents = self.env.agents
        base = self.env.observation_space(self.agents[0]).shape[0]
        return base + (self.num_agents if self.obs_agent_id else 0)

    def _append_agent_id(self, obs, agent_idx):
        if not self.obs_agent_id:
            return obs
        one_hot = np.zeros(self.num_agents, dtype=obs.dtype)
        one_hot[agent_idx] = 1.0
        return np.concatenate([obs, one_hot])

    def _mask_observation(self, obs):
        if self.mask_communication:
            obs = obs.copy()
            obs[-(self.dim_c * (self.num_agents - 1)):] = 0
        return obs

    def _mask_action(self, action):
        if self.mask_communication:
            # action is [movement, communication]
            # set communication to 0 (always the first action)
            action = action.copy() if isinstance(action, np.ndarray) else list(action)
            action[1] = 0
        return action


