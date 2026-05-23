#!/bin/bash
# Test new hyperparameters with a single IQL no-comm run
# If this looks better (lower grad norm, smoother curves), use these for full ablation

cd ..

echo "Testing new hyperparameters (larger buffer, batch size, better lr)..."

uv run python train.py \
    --algorithm iql \
    --mask_communication \
    --experiment_name iql_no_comm_hypertest_v3 \
    --num_timesteps 10000000 \
    --episode_length 25 \
    --lr 3e-4 \
    --gamma 0.99 \
    --batch_size 256 \
    --buffer_size 100000 \
    --min_buffer_size 5000 \
    --epsilon_start 0.5 \
    --epsilon_end 0.05 \
    --epsilon_anneal_time 1000000 \
    --target_update_freq 5000 \
    --num_eval_episodes 30 \
    --eval_interval 50000 \
    --seed 1 \
    --use_wandb

echo "Hyperparameter test complete!"
echo ""
echo "Check wandb for:"
echo "- train/grad_norm should be < 10 (ideally 1-5)"
echo "- train/loss should decrease over time and stabilize < 5"
echo "- eval/average_episode_rewards should be smoother with 30 eval episodes"
echo "- Compare to your previous iql_no_comm runs to see improvement"
