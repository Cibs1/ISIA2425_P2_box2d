from stable_baselines3 import PPO, A2C, DDPG, SAC
import gymnasium as gym
import matplotlib.pyplot as plt
import os

# Ensure TensorFlow does not use GPU (useful for CPU-only algorithms like PPO and A2C)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Define the algorithms and their model paths
algorithms = {
    "PPO": "models/PPO/2900000.zip",
    "A2C": "models/A2C/2900000.zip",
    "DDPG": "models/DDPG/2900000.zip",
    "SAC": "models/SAC/2900000.zip"
}

# Test environment
env_id = 'BipedalWalker-v3'

def evaluate_model(algo, path, n_episodes=3):
    """
    Evaluate a single model on the environment.

    Parameters:
    - algo: Algorithm name (e.g., 'PPO', 'A2C', 'DDPG', 'SAC').
    - path: Path to the saved model file.
    - n_episodes: Number of episodes to evaluate.

    Returns:
    - Average reward over the specified number of episodes.
    """
    print(f"Evaluating {algo}...")

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

            # Optionally stop the episode early to avoid excessive steps
            if steps > 2000:  # Example limit
                print(f"Terminating episode early for {algo} (exceeded 2000 steps).")
                break

        total_rewards.append(episode_reward)

    env.close()
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"{algo}: Average reward = {avg_reward}")
    return avg_reward

# Evaluate each algorithm and store results
results = {}
n_episodes = 5  # Number of episodes for evaluation

for algo, path in algorithms.items():
    try:
        avg_reward = evaluate_model(algo, path, n_episodes=n_episodes)
        results[algo] = avg_reward
    except Exception as e:
        print(f"Error evaluating {algo}: {e}")
        results[algo] = None

# Print evaluation results
print("\nFinal Evaluation Results:")
for algo, reward in results.items():
    if reward is not None:
        print(f"{algo}: {reward:.2f}")
    else:
        print(f"{algo}: Evaluation failed.")

# Plot results
plt.bar(results.keys(), [v if v is not None else 0 for v in results.values()])
plt.xlabel("Algorithm")
plt.ylabel("Average Reward")
plt.title("Algorithm Performance on BipedalWalker-v3")

# Save the plot instead of showing it interactively
plot_filename = "evaluation_results.png"
plt.savefig(plot_filename)
print(f"Plot saved as {plot_filename}")
