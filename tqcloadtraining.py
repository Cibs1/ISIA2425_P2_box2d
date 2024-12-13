import gymnasium as gym
from sb3_contrib import TQC
import os

# Directories for saving models and logs
models_dir = "models/TQC_hardcore"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Create the environment
env = gym.make('BipedalWalker-v3', hardcore=True)
env.reset()

# Load the existing model or initialize a new one
model_path = "1700000.zip"  # Path to the saved model
if os.path.exists(model_path):
    print(f"Loading model from {model_path}...")
    model = TQC.load(model_path, env=env, tensorboard_log=logdir)
else:
    print("No existing model found. Initializing a new model...")
    model = TQC('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

# Training parameters
TIMESTEPS = 100000
starting_iter = 17  # Starting iteration based on 1,700,000 steps

# Training loop
for i in range(starting_iter + 1, starting_iter + 34                                                                                                                                                                                                                                                                                     ):  # Continue training for additional iterations
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="TQC_hardcore")
    model.save(f"{models_dir}/{TIMESTEPS * i}")
    print(f"Model saved at iteration {i} with {TIMESTEPS * i} timesteps.")

print("Training complete!")
