import argparse
import numpy as np
import torch
import wandb
from envs.mpe_wrapper import SimpleReferenceWrapper
from agents.iql import IQLAgent
from agents.vdn import VDNAgent
from utils.buffer import EpisodeReplayBuffer


def get_config():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train IQL/VDN on MPE')

    # Environment args
    parser.add_argument('--env_name', type=str, default='simple_reference', help='MPE environment')
    parser.add_argument('--mask_communication', action='store_true', help='Mask communication for no-comm baseline')

    # Algorithm args
    parser.add_argument('--algorithm', type=str, default='iql', choices=['iql', 'vdn'], help='Algorithm to use')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden layer dimension')
    parser.add_argument('--rnn_hidden_dim', type=int, default=64, help='RNN hidden dimension')

    # Training args
    parser.add_argument('--num_timesteps', type=int, default=1000000, help='Number of training episodes')
    parser.add_argument('--episode_length', type=int, default=25, help='Max episode length')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--buffer_size', type=int, default=5000, help='Replay buffer size')
    parser.add_argument('--min_buffer_size', type=int, default=100, help='Min buffer size before training')

    # Exploration args
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='Starting epsilon')
    parser.add_argument('--epsilon_end', type=float, default=0.05, help='Final epsilon')
    parser.add_argument('--epsilon_anneal_time', type=int, default=500000, help='Timesteps to anneal epsilon')

    # Target network args
    parser.add_argument('--target_update_freq', type=int, default=200, help='Target network update frequency (episodes)')

    # Evaluation args
    parser.add_argument('--num_eval_episodes', type=int, default=10, help='Number of episodes for evaluation')
    parser.add_argument('--eval_interval', type=int, default=50000, help='Evaluate every N timesteps')

    # Logging args
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--wandb_project', type=str, default='comm-vs-ctde', help='Wandb project name')
    parser.add_argument('--experiment_name', type=str, default='debug', help='Experiment name')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')

    args = parser.parse_args()
    return args


def collect_episode(env, agent, epsilon, max_steps):
    """
    Runs a single episode to completion

    Args:
    - env: The environment to run the episode in
    - agent: The agent to use for the episode
    - epsilon: The epsilon value for the episode
    - max_steps: The maximum number of steps to run the episode for

    Returns:
    - episode_data: A dictionary containing the episode data.
        {
            'obs': [],
            'actions': [],
            'rewards': [],
            'next_obs': [],
            'dones': [],
            'hidden_states': []
        }
    """
    obs_list = env.reset()

    obs_buffer = []
    actions_buffer = []
    rewards_buffer = []
    next_obs_buffer = []
    dones_buffer = []
    hidden_states_buffer = []


    # init hidden states for each agent (shape: (num_layers, rnn_hidden_dim))
    hidden_states = [agent.q_network.init_hidden(1).squeeze(1).numpy() for _ in range(env.num_agents)]

    episode_reward = 0

    for timestep in range(max_steps):
        actions = []
        new_hidden_states = []
        # Collect the joint action at the current timestep
        for i, agent_obs in enumerate(obs_list):
            action, new_h = agent.select_action(agent_obs, hidden_states[i], epsilon)
            actions.append(action)
            new_hidden_states.append(new_h)

        # Take action
        next_obs_list, reward_list, done_list, info_list = env.step(actions)

        # Store transistion
        obs_buffer.append(obs_list)
        actions_buffer.append(actions)
        rewards_buffer.append(reward_list)
        next_obs_buffer.append(next_obs_list)
        dones_buffer.append(done_list)
        hidden_states_buffer.append(hidden_states)

        # Track episode reward
        episode_reward += sum(reward_list)

        # Update obs and hidden state
        obs_list = next_obs_list
        hidden_states = new_hidden_states

        if any(done_list):
            break

    episode_data = {
        'obs': np.array(obs_buffer),
        'actions': np.array(actions_buffer),
        'rewards': np.array(rewards_buffer),
        'next_obs': np.array(next_obs_buffer),
        'dones': np.array(dones_buffer),
        'hidden_states': np.array(hidden_states_buffer),
    }

    return episode_data, episode_reward


def evaluate(env, agent, num_episodes, max_steps):
    """
    Evaluate the agent greedily (no exploration) for multiple episodes

    Args:
        env: The environment
        agent: The agent to evaluate
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode

    Returns:
        eval_rewards: List of episode rewards
    """
    eval_rewards = []

    for _ in range(num_episodes):
        _, episode_reward = collect_episode(env, agent, epsilon=0.0, max_steps=max_steps)
        eval_rewards.append(episode_reward)

    return eval_rewards


def main():
    args = get_config()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Training {args.algorithm} on {args.env_name}, mask_comm={args.mask_communication}, seed={args.seed}")

    # Initialize environment
    env = SimpleReferenceWrapper(mask_communication=args.mask_communication)
    obs_dim = env.get_obs_dim()
    action_dims = env.get_action_dim()
    epsilon = args.epsilon_start

    if args.algorithm == 'iql':
        # Initialize IQL agent
        agent = IQLAgent(
            obs_dim=obs_dim,
            action_dims=action_dims,
            hidden_dim=args.hidden_dim,
            rnn_hidden_dim=args.rnn_hidden_dim,
            lr=args.lr,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
        )
    elif args.algorithm == 'vdn':
      agent = VDNAgent(
          obs_dim=obs_dim,
          action_dims=action_dims,
          hidden_dim=args.hidden_dim,
          rnn_hidden_dim=args.rnn_hidden_dim,
          lr=args.lr,
          gamma=args.gamma,
          epsilon_start=args.epsilon_start,
          epsilon_end=args.epsilon_end,
      )

    # Initialize buffer
    buffer = EpisodeReplayBuffer(max_capacity=args.buffer_size)

    # Optionally, init WandB
    if (args.use_wandb):
        wandb.init(
            project=args.wandb_project,
            name=f"{args.algorithm}_{args.experiment_name}_seed{args.seed}",
            config=vars(args)
        )

    episode_rewards = []
    total_timesteps = 0
    log_interval = 10000  # Log every 10k timesteps
    last_eval_step = 0

    # Training loop
    while total_timesteps < args.num_timesteps:
        # Collect episode
        episode_data, episode_reward = collect_episode(env, agent, epsilon, args.episode_length)
        episode_rewards.append(episode_reward)

        # Track timesteps
        episode_length = len(episode_data['obs'])
        total_timesteps += episode_length * env.num_agents  # Count timesteps for all agents

        if total_timesteps % log_interval < (episode_length * env.num_agents):  # Just crossed logging boundary
            print(f"Timesteps: {total_timesteps}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}, Buffer: {len(buffer)}")

        # Add episode to buffer
        buffer.add_episode(**episode_data)

        # Train if buffer is full enough
        if len(buffer) >= args.min_buffer_size:
            # Sample batch
            batch = buffer.sample(args.batch_size)

            loss, grad_norm, mean_q = agent.train(batch)

            if total_timesteps % log_interval < (episode_length * env.num_agents) and args.use_wandb:
                recent_rewards = episode_rewards[-100:]
                avg_reward = np.mean(recent_rewards)
                std_reward = np.std(recent_rewards)
                min_reward = np.min(recent_rewards)
                max_reward = np.max(recent_rewards)

                # Log with organized prefixes (matching off-policy style)
                wandb.log({
                    # Training metrics
                    'train/loss': loss,
                    'train/grad_norm': grad_norm,
                    'train/Q_tot': mean_q,

                    # Environment metrics
                    'env/average_episode_rewards': avg_reward,
                    'env/episode_reward_std': std_reward,
                    'env/episode_reward_min': min_reward,
                    'env/episode_reward_max': max_reward,

                    # Exploration
                    'train/epsilon': epsilon,
                }, step=total_timesteps)  # Use timesteps as x-axis

        # Periodically update the target network (based on episodes for now)
        if total_timesteps % args.target_update_freq == 0 and total_timesteps > 0:
            agent.update_target_network()

        # Linear epsilon decay based on timesteps
        epsilon_progress = min(1.0, total_timesteps / args.epsilon_anneal_time)
        epsilon = args.epsilon_start - (args.epsilon_start - args.epsilon_end) * epsilon_progress

        # Periodic evaluation
        if total_timesteps - last_eval_step >= args.eval_interval:
            print(f"Running evaluation at timestep {total_timesteps}...")
            eval_rewards = evaluate(env, agent, args.num_eval_episodes, args.episode_length)
            eval_avg_reward = np.mean(eval_rewards)
            eval_std_reward = np.std(eval_rewards)
            eval_min_reward = np.min(eval_rewards)
            eval_max_reward = np.max(eval_rewards)

            print(f"Eval: avg={eval_avg_reward:.2f}, std={eval_std_reward:.2f}, min={eval_min_reward:.2f}, max={eval_max_reward:.2f}")

            if args.use_wandb:
                wandb.log({
                    'eval/average_episode_rewards': eval_avg_reward,
                    'eval/episode_reward_std': eval_std_reward,
                    'eval/episode_reward_min': eval_min_reward,
                    'eval/episode_reward_max': eval_max_reward,
                }, step=total_timesteps)

            last_eval_step = total_timesteps

    if args.use_wandb:
        wandb.finish()
    print("Training complete!")



if __name__ == "__main__":
    main()
