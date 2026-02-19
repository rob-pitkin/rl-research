#!/bin/bash
# Train VDN with communication

cd ..

uv run python train.py \
    --algorithm vdn \
    --experiment_name vdn_with_comm \
    --num_timesteps 2000000 \
    --episode_length 25 \
    --lr 3e-4 \
    --gamma 0.99 \
    --batch_size 32 \
    --buffer_size 5000 \
    --min_buffer_size 100 \
    --epsilon_start 1.0 \
    --epsilon_end 0.05 \
    --epsilon_anneal_time 500000 \
    --target_update_freq 200 \
    --num_eval_episodes 10 \
    --eval_interval 50000 \
    --seed 1 \
    --use_wandb

echo "Training complete!"
