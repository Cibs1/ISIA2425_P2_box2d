from sb3_contrib import TRPO
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import os

# Define directories
model_dir = "models/TRPO"  # Directory containing TRPO zip files
log_dir_base = "logs/TRPO_0"  # Base directory for log files
os.makedirs(log_dir_base, exist_ok=True)

# Define the evaluation settings
steps = [1000000, 2000000, 3000000]  # Training steps to evaluate
n_episodes = 5  # Number of episodes per evaluation
max_steps = 2000  # Maximum steps per episode to prevent long-running episodes

# Test environment
env_id = 'BipedalWalker-v3'

def evaluate_trpo_model(path, n_episodes=5, max_steps=2000):
    """
    Evaluate a TRPO model and return average reward and steps.

    Parameters:
    - path: Path to the TRPO model file.
    - n_episodes: Number of episodes to evaluate.
    - max_steps: Maximum steps per episode.

    Returns:
    - Average reward and average steps over the episodes.
    """
    print(f"Evaluating TRPO model from {path}...")

    # Load the model
    model = TRPO.load(path, device="cpu")

    # Create the environment
    env = gym.make(env_id)
    total_rewards = []
    total_steps = []

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

            if steps >= max_steps:
                print(f"Terminating episode early (exceeded {max_steps} steps).")
                break

        total_rewards.append(episode_reward)
        total_steps.append(steps)

    env.close()

    avg_reward = sum(total_rewards) / len(total_rewards)
    avg_steps = sum(total_steps) / len(total_steps)

    print(f"Average Reward: {avg_reward}, Average Steps: {avg_steps}")
    return avg_reward, avg_steps

# Evaluate models and create separate log files
for step in steps:
    model_path = os.path.join(model_dir, f"{step}.zip")

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Skipping.")
        continue

    # Log directory for this iteration
    log_dir = os.path.join(log_dir_base, f"iteration_{step}")
    os.makedirs(log_dir, exist_ok=True)

    # Create a new SummaryWriter for this iteration
    writer = SummaryWriter(log_dir)

    # Evaluate the model
    avg_reward, avg_steps = evaluate_trpo_model(model_path, n_episodes=n_episodes, max_steps=max_steps)

    # Log the metrics
    writer.add_scalar("average_reward", avg_reward, step)
    writer.add_scalar("steps", avg_steps, step)

    # Close the writer
    writer.close()

    print(f"Logs for step {step} saved to {log_dir}")

print("All iterations processed.")
