import subprocess
import argparse
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_dir():
  script_dir = Path(__file__).parent.resolve()  # experiments/
  project_dir = script_dir.parent                # llm-credit-assignment/
  epymarl_dir = project_dir / "epymarl"

  if not epymarl_dir.exists():
    print(f"Error: EPyMARL directory not found at {epymarl_dir}")
    sys.exit(1)

  print(f"EPyMARL directory: {epymarl_dir}")
  return epymarl_dir

def run_experiment(config, env, env_args, t_max, seed, epymarl_dir):
  cmd = [
    "uv",
    "run",
    "python3",
    "src/main.py",
    f"--config={config}",
    f"--env-config={env}",
    "with",
    f"env_args.time_limit={env_args['time_limit']}",
    f"env_args.key={env_args['key']}",
    f"t_max={t_max}",
    f"seed={seed}"
  ]
  result = subprocess.run(cmd, cwd=epymarl_dir)
  if result.returncode == 0:
    print(f"✓ Seed {seed} completed successfully")
    return True
  else:
    print(f"✗ Seed {seed} failed with return code {result.returncode}")
    return False

def run_multiple_seeds(config, env, env_args, t_max, seeds, epymarl_dir, parallel=1):
  results = []
  start_time = time.time()
  if parallel == 1:
    for seed in range(1, seeds + 1):
      print(f"\n{'='*60}")
      print(f"Running seed {seed}/{seeds}")
      print(f"{'='*60}\n")

      success = run_experiment(config, env, env_args, t_max, seed, epymarl_dir)
      results.append((seed, success))

  else:
    with ProcessPoolExecutor(max_workers=parallel) as executor:
      futures = {
        executor.submit(run_experiment, config, env, env_args, t_max, seed, epymarl_dir) : seed
        for seed in range(1, seeds + 1)
      }

      for future in as_completed(futures):
        seed = futures[future]
        success = future.result()
        results.append((seed, success))
        print(f"Seed {seed} completed")

  elapsed_time = time.time() - start_time
  print(f"\nTotal time: {elapsed_time/60:.1f} minutes")
  print("\n\nExperiment Summary:")
  successes = sum(1 for _, success in results if success)
  print(f"Completed: {successes}/{len(results)} seeds")
  for seed, success in results:
    print(f"  Seed {seed}: {'✓ Success' if success else '✗ Failure'}")

  return results

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str, required=True, help="Algorithm config (iql, vdn, qmix, etc.)")
  parser.add_argument("--env", type=str, default="gymma", help="Environment type")
  parser.add_argument("--env-key", type=str, required=True, help="Environment key")
  parser.add_argument("--time-limit", type=int, default=50, help="Episode time limit")
  parser.add_argument("--t-max", type=int, required=True, help="Total timesteps")
  parser.add_argument("--seeds", type=int, required=True, help="Number of seeds")
  parser.add_argument("--parallel", type=int, default=1, help="Number of seeds to run in parallel")
  args = parser.parse_args()

  env_args = {}
  if args.env_key:
    env_args["key"] = args.env_key
  if args.time_limit:
    env_args["time_limit"] = args.time_limit

  print(f"Args: {args}, Env args: {env_args}")

  epymarl_dir = get_dir()

  run_multiple_seeds(args.config, args.env, env_args, args.t_max, args.seeds, epymarl_dir, args.parallel)



if __name__ == "__main__":
  main()
