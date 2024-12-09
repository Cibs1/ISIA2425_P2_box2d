import matplotlib.pyplot as plt

# Sample data (replace with actual extracted data)
ppo_rewards = [100, 200, 300, 400]  # Replace with extracted rewards for PPO
a2c_rewards = [90, 190, 280, 350]   # Replace with extracted rewards for A2C
ddpg_rewards = [80, 180, 260, 330]  # Replace with extracted rewards for DDPG
sac_rewards = [110, 210, 310, 380]  # Replace with extracted rewards for SAC
iterations = [100000, 200000, 300000, 400000]  # Iteration steps

# Plot
plt.plot(iterations, ppo_rewards, label='PPO')
plt.plot(iterations, a2c_rewards, label='A2C')
plt.plot(iterations, ddpg_rewards, label='DDPG')
plt.plot(iterations, sac_rewards, label='SAC')

# Add labels and legend
plt.xlabel('Iterations')
plt.ylabel('Reward')
plt.title('Algorithm Performance Comparison')
plt.legend()
plt.show()
