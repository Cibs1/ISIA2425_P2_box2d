from stable_baselines3 import PPO, A2C, DDPG, SAC
from sb3_contrib import TRPO
import gymnasium as gym
import matplotlib.pyplot as plt
import os
import time

# Ensure TensorFlow does not use GPU (useful for CPU-only algorithms like PPO and A2C)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Define the algorithms and their model paths
algorithms = {
    "PPO": {"path": "models/PPO", "steps": [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000, 2000000, 2100000, 2200000, 2300000, 2400000, 2500000, 2600000, 2700000, 2800000, 2900000]},
    "A2C": {"path": "models/A2C", "steps": [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000, 2000000, 2100000, 2200000, 2300000, 2400000, 2500000, 2600000, 2700000, 2800000, 2900000]},
    "DDPG": {"path": "models/DDPG", "steps": [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000, 2000000, 2100000, 2200000, 2300000, 2400000, 2500000, 2600000, 2700000, 2800000, 2900000]},
    "SAC": {"path": "models/SAC", "steps": [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000, 2000000, 2100000, 2200000, 2300000, 2400000, 2500000, 2600000, 2700000, 2800000, 2900000]},
    "TRPO": {"path": "models/TRPO", "steps": [1000000, 2000000, 3000000]}  # Add TRPO model paths and steps
}

# Test environment
env_id = 'BipedalWalker-v3'

def evaluate_model(algo, path, n_episodes=3, max_steps=2000):
    """
    Evaluate a single model on the environment.

    Parameters:
    - algo: Algorithm name (e.g., 'PPO', 'A2C', 'DDPG', 'SAC', 'TRPO').
    - path: Path to the saved model file.
    - n_episodes: Number of episodes to evaluate.
    - max_steps: Maximum steps per episode to prevent infinite loops.

    Returns:
    - Average reward over the specified number of episodes.
    """
    print(f"Evaluating {algo} from {path}...")

    # Force CPU usage for PPO and A2C
    device = 'cpu' if algo in ["PPO", "A2C"] else 'auto'
    model = eval(algo).load(path, device=device)

    env = gym.make(env_id)
    total_rewards = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            # Terminate if max steps reached
            if steps >= max_steps:
                print(f"Terminating episode early for {algo} (exceeded {max_steps} steps).")
                break

        total_rewards.append(episode_reward)

    env.close()
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"{algo}: Average reward = {avg_reward}")
    return avg_reward

# Collect data for plotting
plot_data = {}

for algo, config in algorithms.items():
    print(f"Processing {algo}...")
    step_ratios = []
    for step in config["steps"]:
        model_path = f"{config['path']}/{step}.zip"
        try:
            start_time = time.time()
            reward = evaluate_model(algo, model_path, n_episodes=5)
            elapsed_time = time.time() - start_time
            ratio = elapsed_time / reward if reward != 0 else float('inf')
            step_ratios.append((step, ratio))
            print(f"Step {step}: Time = {elapsed_time:.2f}s, Reward = {reward:.2f}, Ratio = {ratio:.4f}")
        except Exception as e:
            print(f"Error processing {algo} at step {step}: {e}")
            step_ratios.append((step, None))

    plot_data[algo] = step_ratios

# Plot the results
plt.figure(figsize=(10, 6))

for algo, data in plot_data.items():
    steps, ratios = zip(*[(step, ratio) for step, ratio in data if ratio is not None])
    plt.plot(steps, ratios, label=algo)

plt.xlabel("Training Steps")
plt.ylabel("Time Taken / Reward")
plt.title("Time-to-Reward Ratio vs Training Steps")
plt.legend()
plt.grid()
plt.tight_layout()

# Save the plot
plot_filename = "time_to_reward_ratio.png"
plt.savefig(plot_filename)
print(f"Plot saved as {plot_filename}")
