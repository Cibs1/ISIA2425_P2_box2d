from bipedal_env2 import BipedalWalker

env = BipedalWalker(render_mode="human", hardcore=True)
obs = env.reset()

for step in range(1000):
    env.render()
    action = env.action_space.sample()  # Random actions to test terrain
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs = env.reset()

env.close()
