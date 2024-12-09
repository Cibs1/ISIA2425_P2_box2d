import gymnasium as gym
from sb3_contrib import TQC
import os

# Directories for saving models and logs
models_dir = "models/TQC"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Create the environment
env = gym.make('BipedalWalker-v3')
env.reset()

# Initialize the model with DDPG
model = TQC('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

# Training parameters
TIMESTEPS = 100000
iters = 0

# Training loop
for i in range(50):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="TQC")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
