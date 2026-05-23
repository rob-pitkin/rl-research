#!/bin/bash
# Run full 2x2 ablation: (IQL vs VDN) x (no-comm vs with-comm) with 5 seeds each
# Total: 20 runs

cd ..

# Configuration
algorithms=("iql" "vdn")
seeds=(1 2 3 4 5)
max_parallel=2  # Max number of parallel runs

# Function to run a single training job
run_training() {
    local algo=$1
    local comm_flag=$2
    local comm_name=$3
    local seed=$4

    echo "Starting: ${algo}_${comm_name}_seed${seed}"

    uv run python train.py \
        --algorithm $algo \
        $comm_flag \
        --experiment_name ${algo}_${comm_name} \
        --num_timesteps 10000000 \
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
        --seed $seed \
        --use_wandb

    echo "Completed: ${algo}_${comm_name}_seed${seed}"
}

# Export function so parallel processes can use it
export -f run_training

# Create array of all jobs
jobs=()
for algo in "${algorithms[@]}"; do
    # No communication runs
    for seed in "${seeds[@]}"; do
        jobs+=("$algo --mask_communication no_comm $seed")
    done
    # With communication runs
    for seed in "${seeds[@]}"; do
        jobs+=("$algo \"\" with_comm $seed")
    done
done

echo "Total jobs to run: ${#jobs[@]}"
echo "Running with max $max_parallel parallel jobs"
echo ""

# Run jobs in parallel with limited parallelism
printf '%s\n' "${jobs[@]}" | xargs -n 4 -P $max_parallel bash -c 'run_training "$@"' _

echo ""
echo "All training runs complete!"
