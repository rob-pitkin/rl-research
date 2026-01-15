"""
Exploration script for Level-Based Foraging environment.
Runs random agents and logs detailed information about:
- State representations
- Action space
- Reward structure
- Episode statistics
"""

import gymnasium as gym
import lbforaging
import numpy as np
from collections import defaultdict
from trajectory_to_text import LBFTranslator


def decode_observation(obs, agent_id, n_food=2, n_agents=2):
    """
    Decode LBF observation vector into human-readable format.

    IMPORTANT: In LBF, each agent's observation is agent-centric:
    - The observing agent always appears first in the agent list
    - Food order is consistent across agents (based on grid position)

    Format: [food1_y, food1_x, food1_level, food2_y, food2_x, food2_level,
             self_y, self_x, self_level, other_y, other_x, other_level]
    """
    obs_dict = {}
    idx = 0

    # Decode food items first
    for i in range(n_food):
        y = int(obs[idx])
        x = int(obs[idx + 1])
        level = int(obs[idx + 2])

        # -1 indicates no food (already collected)
        if y == -1:
            obs_dict[f'food_{i}'] = None
        else:
            obs_dict[f'food_{i}'] = {'y': y, 'x': x, 'level': level}
        idx += 3

    # Decode agents - "self" comes first, then others
    for i in range(n_agents):
        actual_agent_id = agent_id if i == 0 else (1 - agent_id)  # For 2 agents
        obs_dict[f'agent_{actual_agent_id}'] = {
            'y': int(obs[idx]),
            'x': int(obs[idx + 1]),
            'level': int(obs[idx + 2]),
        }
        idx += 3

    return obs_dict


def action_to_string(action):
    """Convert action integer to human-readable string."""
    action_map = {
        0: "NOOP",
        1: "NORTH",
        2: "SOUTH",
        3: "WEST",
        4: "EAST",
        5: "LOAD",  # Attempt to pick up food
    }
    return action_map.get(action, f"UNKNOWN_{action}")


def run_episode(env, episode_num, verbose=True):
    """Run a single episode with random actions and log transitions."""
    obs, info = env.reset()

    print(f"\n=== DEBUG: Raw observations at reset ===")
    print(f"obs type: {type(obs)}, length: {len(obs)}")
    print(f"Agent 0 obs shape: {obs[0].shape}, values: {obs[0]}")
    print(f"Agent 1 obs shape: {obs[1].shape}, values: {obs[1]}")
    print("=" * 60)

    episode_data = {
        'episode': episode_num,
        'transitions': [],
        'total_reward': [0, 0],  # Per-agent cumulative reward
        'steps': 0,
    }

    done = False
    truncated = False

    if verbose:
        print(f"\n{'='*60}")
        print(f"Episode {episode_num}")
        print(f"{'='*60}")
        print("\nInitial State:")
        for agent_id in range(2):
            obs_dict = decode_observation(obs[agent_id], agent_id)
            print(f"  Agent {agent_id}: {obs_dict}")

    while not (done or truncated):
        # Sample random actions
        actions = env.action_space.sample()

        # Step environment
        next_obs, rewards, done, truncated, info = env.step(actions)

        # Log transition
        transition = {
            'step': episode_data['steps'],
            'obs': [obs[0].copy(), obs[1].copy()],
            'actions': actions,
            'action_strings': [action_to_string(actions[0]), action_to_string(actions[1])],
            'rewards': rewards,
            'next_obs': [next_obs[0].copy(), next_obs[1].copy()],
            'done': done,
        }
        episode_data['transitions'].append(transition)

        # Update cumulative rewards
        episode_data['total_reward'][0] += rewards[0]
        episode_data['total_reward'][1] += rewards[1]
        episode_data['steps'] += 1

        if verbose and (rewards[0] != 0 or rewards[1] != 0):
            print(f"\nStep {episode_data['steps']}:")
            print(f"  Actions: Agent 0: {action_to_string(actions[0])}, Agent 1: {action_to_string(actions[1])}")
            print(f"  Rewards: {rewards}")

        obs = next_obs

    if verbose:
        print(f"\nEpisode {episode_num} Complete:")
        print(f"  Total steps: {episode_data['steps']}")
        print(f"  Agent 0 reward: {episode_data['total_reward'][0]}")
        print(f"  Agent 1 reward: {episode_data['total_reward'][1]}")
        print(f"  Team reward: {sum(episode_data['total_reward'])}")

    return episode_data


def main():
    """Run exploration of LBF environment."""
    print("Level-Based Foraging Environment Exploration")
    print("=" * 60)

    # Create environment
    env_name = "Foraging-8x8-2p-2f-coop-v3"
    env = gym.make(env_name)

    print(f"\nEnvironment: {env_name}")
    print(f"  Grid size: 8x8")
    print(f"  Agents: 2")
    print(f"  Food items: 2")
    print(f"  Cooperative: Yes (agents must cooperate to collect food)")
    print(f"  Observation space per agent: {env.observation_space[0]}")
    print(f"  Action space per agent: {env.action_space[0]}")
    print(f"  Actions: NOOP, MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, LOAD")

    # Run episodes
    num_episodes = 10
    all_episodes = []

    print(f"\n{'='*60}")
    print(f"Running {num_episodes} episodes with random agents")
    print(f"{'='*60}")

    for ep in range(num_episodes):
        episode_data = run_episode(env, ep + 1, verbose=(ep < 3))  # Verbose for first 3
        all_episodes.append(episode_data)

    # Compute statistics
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")

    episode_lengths = [ep['steps'] for ep in all_episodes]
    team_rewards = [sum(ep['total_reward']) for ep in all_episodes]
    agent0_rewards = [ep['total_reward'][0] for ep in all_episodes]
    agent1_rewards = [ep['total_reward'][1] for ep in all_episodes]

    print(f"\nEpisode Lengths:")
    print(f"  Mean: {np.mean(episode_lengths):.2f}")
    print(f"  Min: {np.min(episode_lengths)}")
    print(f"  Max: {np.max(episode_lengths)}")

    print(f"\nTeam Rewards:")
    print(f"  Mean: {np.mean(team_rewards):.2f}")
    print(f"  Min: {np.min(team_rewards)}")
    print(f"  Max: {np.max(team_rewards)}")
    print(f"  Success rate: {sum(1 for r in team_rewards if r > 0) / num_episodes * 100:.1f}%")

    print(f"\nAgent 0 Rewards:")
    print(f"  Mean: {np.mean(agent0_rewards):.2f}")

    print(f"\nAgent 1 Rewards:")
    print(f"  Mean: {np.mean(agent1_rewards):.2f}")

    # Action distribution
    action_counts = defaultdict(int)
    for ep in all_episodes:
        for trans in ep['transitions']:
            action_counts[trans['action_strings'][0]] += 1
            action_counts[trans['action_strings'][1]] += 1

    print(f"\nAction Distribution (all agents, all episodes):")
    total_actions = sum(action_counts.values())
    for action, count in sorted(action_counts.items()):
        print(f"  {action}: {count} ({count/total_actions*100:.1f}%)")

    print(f"\n{'='*60}")
    print("Key Observations for State → Text Translation:")
    print(f"{'='*60}")
    print("""
1. State representation: Each agent observes positions and levels of all agents and food
2. Actions are discrete: 6 possible actions per agent
3. Rewards are sparse: Only when food is collected
4. Cooperation required: In 'coop' environments, multiple agents needed for high-level food
5. Episodes are relatively short: ~50 steps max

For text translation, we should include:
- Agent positions (x, y coordinates)
- Agent levels (capability to pick up food)
- Food positions and levels
- Actions taken by each agent
- Rewards received
    """)

    env.close()

    print(f"\n{'='*60}")
    print("Saving example trajectories for prompt design")
    print(f"{'='*60}")

    translator = LBFTranslator()
    with open("example_trajectories.txt", "w") as f:
        # Save first 5 episodes
        for i, ep in enumerate(all_episodes[:5]):
            f.write(f"\n{'='*80}\n")
            f.write(f"Episode {i+1}\n")
            f.write(f"{'='*80}\n")
            trajectory_text = translator.trajectory_to_text(ep['transitions'], include_all_steps=True)
            f.write(trajectory_text)

    print(f"Saved example trajectories to example_trajectories.txt")


if __name__ == "__main__":
    main()
