import gymnasium as gym
from stable_baselines3 import A2C
import time
import csv
import sys

# Redirect terminal output to both console and CSV
class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.logfile = open(filepath, "w", newline="")
        self.csv_writer = csv.writer(self.logfile)
        self.csv_writer.writerow(["Message"])  # Write header row

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        if message.strip():  # Avoid logging empty lines
            self.csv_writer.writerow([message.strip()])

    def flush(self):
        pass  # Needed for compatibility with sys.stdout

# Set up logging
log_path = "training_output_a2c.csv"
sys.stdout = Logger(log_path)

# Create the environment
env = gym.make('BipedalWalker-v3')

# Create the model and specify GPU usage
model = A2C('MlpPolicy', env, verbose=1, device="cuda")

# Training settings
total_timesteps = 1_000_000  # Total timesteps to train the model
save_interval = 10_000       # Save the model every 10,000 timestepsz
model_save_path = "a2c_bipedalwalker"  # File path for saving the model

# Start training
start_time = time.time()
print("Starting training...")

for step in range(0, total_timesteps, save_interval):
    print(f"Training from timestep {step} to {step + save_interval}...")
    model.learn(total_timesteps=save_interval)
    
    # Save the model periodically
    model.save(model_save_path)
    print(f"Model saved at timestep {step + save_interval}.")

# Final save after training
model.save(f"{model_save_path}_final")
print(f"Training completed in {time.time() - start_time:.2f} seconds.")
print(f"Final model saved as {model_save_path}.zip")

# Test the trained model
obs, _ = env.reset()  # Fix: Unpack the tuple returned by `env.reset()`
for step in range(200):
    env.render()
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()  # Fix: Unpack the tuple here as well

env.close()

# Close logging
sys.stdout.logfile.close()
sys.stdout = sys.stdout.terminal
