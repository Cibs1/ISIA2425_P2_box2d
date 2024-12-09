from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

# Load TensorBoard logs
log_dir = "logs/SAC_0"  # Replace with each algorithm's log folder
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()

# Extract rewards or other metrics
rewards = ea.Scalars('rollout/ep_rew_mean')  # Replace with appropriate metric key
rewards_df = pd.DataFrame([(e.step, e.value) for e in rewards], columns=['step', 'reward'])

# Save or process the data
print(rewards_df)
