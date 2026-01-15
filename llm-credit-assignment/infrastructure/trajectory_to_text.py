"""
State-to-text translation for Level-Based Foraging environment.

Provides deterministic conversion from LBF observations, actions, and rewards
to natural language descriptions suitable for LLM processing.
"""

import numpy as np
from typing import List, Dict, Tuple, Any


class LBFTranslator:
    """Translates LBF environment state/action/reward to natural language."""

    def __init__(self, num_agents=2, num_food=2, grid_size=(8, 8)):
        self.num_agents = num_agents
        self.num_food = num_food
        self.grid_size = grid_size

        self.action_names = {
            0: "waited",
            1: "moved north",
            2: "moved south",
            3: "moved west",
            4: "moved east",
            5: "attempted to load food",
        }

    def decode_observation(self, obs: np.ndarray) -> Dict[str, Any]:
        """
        Decode LBF observation vector into structured dictionary.

        Args:
            obs: Observation vector [agent0_x, agent0_y, agent0_level, ..., food0_x, food0_y, food0_level, ...]

        Returns:
            Dictionary with 'agents' and 'food' keys
        """
        idx = 0
        state = {'agents': [], 'food': []}

        # Decode food
        for i in range(self.num_food):
            food = {
                'id': i,
                'x': int(obs[idx + 1]),
                'y': int(obs[idx]),
                'level': int(obs[idx + 2]),
            }
            # Check if food exists (x, y >= 0)
            if food['x'] >= 0 and food['y'] >= 0:
                state['food'].append(food)
            idx += 3

        # Decode agents
        for i in range(self.num_agents):
            agent = {
                'id': i,
                'x': int(obs[idx + 1]),
                'y': int(obs[idx]),
                'level': int(obs[idx + 2]),
            }
            state['agents'].append(agent)
            idx += 3

        return state

    def state_to_text(self, obs: np.ndarray, agent_id: int = None) -> str:
        """
        Convert observation to natural language description.

        Args:
            obs: Observation vector
            agent_id: Which agent's perspective (optional, defaults to global view)

        Returns:
            Natural language description of the state
        """
        state = self.decode_observation(obs)

        lines = []

        # Describe agents
        for agent in state['agents']:
            if agent_id is not None and agent['id'] == agent_id:
                lines.append(f"You (Agent {agent['id']}) are at row {agent['y']}, column {agent['x']} with level {agent['level']}.")
            else:
                lines.append(f"Agent {agent['id']} is at row {agent['y']}, column {agent['x']} with level {agent['level']}.")

        return " ".join(lines)

    def action_to_text(self, action: int, agent_id: int) -> str:
        """
        Convert action to natural language.

        Args:
            action: Action integer (0-5)
            agent_id: Agent taking the action

        Returns:
            Natural language description of the action
        """
        action_name = self.action_names.get(action, f"performed unknown action {action}")
        return f"Agent {agent_id} {action_name}"

    def joint_action_to_text(self, actions: Tuple[int, ...]) -> str:
        """
        Convert joint actions (all agents) to natural language.

        Args:
            actions: Tuple of actions, one per agent

        Returns:
            Natural language description of all actions
        """
        action_texts = [self.action_to_text(actions[i], i) for i in range(len(actions))]
        return ". ".join(action_texts) + "."

    def reward_to_text(self, rewards: List[float], total_reward: float = None) -> str:
        """
        Convert rewards to natural language.

        Args:
            rewards: List of rewards, one per agent
            total_reward: Optional total team reward

        Returns:
            Natural language description of rewards
        """
        if total_reward is None:
            total_reward = sum(rewards)

        lines = []
        if total_reward == 0:
            lines.append("No food was collected.")
        else:
            lines.append(f"Food was collected! Total team reward: {total_reward}.")

        for i, reward in enumerate(rewards):
            if reward > 0:
                lines.append(f"Agent {i} received reward {reward}.")

        return " ".join(lines)

    def transition_to_text(
        self,
        obs: np.ndarray,
        actions: Tuple[int, ...],
        rewards: List[float],
        next_obs: np.ndarray,
        step: int,
    ) -> str:
        """
        Convert a full transition (s, a, r, s') to natural language.

        Args:
            obs: Current observation
            actions: Actions taken by all agents
            rewards: Rewards received by all agents
            next_obs: Next observation
            step: Step number

        Returns:
            Natural language description of the transition
        """
        lines = [
            f"Step {step}:",
            f"State: {self.state_to_text(obs)}",
            f"Actions: {self.joint_action_to_text(actions)}",
            f"Outcome: {self.reward_to_text(rewards)}",
        ]

        # Only describe next state if significantly different or reward received
        if sum(rewards) > 0:
          # Describe what changed - specifically which food was collected
          next_state = self.decode_observation(next_obs)
          if next_state['food']:  # if any food remains
              food_desc = ". ".join([
                  f"Food {f['id']} remains at row {f['y']}, column {f['x']} with level {f['level']}"
                  for f in next_state['food']
              ])
              lines.append(f"Remaining food: {food_desc}.")
          else:
              lines.append("All food has been collected.")

        return "\n  ".join(lines)

    def trajectory_to_text(
        self,
        transitions: List[Dict[str, Any]],
        include_all_steps: bool = False,
    ) -> str:
        """
        Convert entire episode trajectory to natural language.

        Args:
            transitions: List of transition dictionaries with keys 'obs', 'actions', 'rewards', 'next_obs', 'step'
            include_all_steps: If False, only include steps with non-zero rewards

        Returns:
            Natural language description of the trajectory
        """
        lines = ["Episode Trajectory:"]

        # Initial state
        if transitions:
            initial_state = self.state_to_text(transitions[0]['obs'][0])  # obs[0] for agent 0's view
            lines.append(f"Initial state: {initial_state}")
            initial_food_state = self.decode_observation(transitions[0]['obs'][0])
            if initial_food_state['food']:
                food_desc = ". ".join([
                    f"Food {f['id']} is at row {f['y']}, column {f['x']} with level {f['level']}"
                    for f in initial_food_state['food']
                ])
                lines.append(f"Initial food: {food_desc}.")

        # Transitions
        for trans in transitions:
            # Skip steps with no reward unless include_all_steps=True
            if not include_all_steps and sum(trans['rewards']) == 0:
                continue

            trans_text = self.transition_to_text(
                obs=trans['obs'][0],  # Use agent 0's observation
                actions=trans['actions'],
                rewards=trans['rewards'],
                next_obs=trans['next_obs'][0],
                step=trans['step'],
            )
            lines.append(trans_text)

        # Summary
        total_reward = sum(sum(trans['rewards']) for trans in transitions)
        total_steps = len(transitions)
        lines.append(f"\nEpisode Summary: {total_steps} steps, total team reward: {total_reward}")

        return "\n\n".join(lines)


# Example usage
if __name__ == "__main__":
    translator = LBFTranslator()

    # Example observation (2 agents, 2 food items)
    obs = np.array([
        1.0, 4.0, 2.0,  # Agent 0: x=1, y=4, level=2
        4.0, 1.0, 2.0,  # Agent 1: x=4, y=1, level=2
        5.0, 6.0, 1.0,  # Food 0: x=5, y=6, level=1
        4.0, 3.0, 1.0,  # Food 1: x=4, y=3, level=1
    ])

    print("State to Text:")
    print(translator.state_to_text(obs))
    print()

    print("Action to Text:")
    actions = (4, 1)  # Agent 0: move right, Agent 1: move up
    print(translator.joint_action_to_text(actions))
    print()

    print("Reward to Text:")
    rewards = [0.5, 0.5]
    print(translator.reward_to_text(rewards))
    print()

    print("Transition to Text:")
    next_obs = obs.copy()
    next_obs[0] += 1  # Agent 0 moved right
    print(translator.transition_to_text(obs, actions, rewards, next_obs, step=1))
