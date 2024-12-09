from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from sb3_contrib import TRPO
from sb3_contrib import TQC
import gymnasium as gym

def test_agent(model_path, steps=1000):
    # Load the trained model
    model = TQC.load(model_path)

    # Create the environment
    env = gym.make('BipedalWalker-v3',render_mode="human")

    # Reset the environment
    obs, _ = env.reset()

    # Test the model
    for step in range(steps):
        env.render()
        # Use the trained model to predict actions
        action, _ = model.predict(obs)
        # Step through the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Print step information
        print(f"Step: {step}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

        # Reset the environment if an episode ends
        if terminated or truncated:
            obs, _ = env.reset()

    # Close the environment
    env.close()

if __name__ == "__main__":
    # Path to your trained model
    model_path = "models/TQC/1000000.zip"  # Update with your file path
    test_agent(model_path)
