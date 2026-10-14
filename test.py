import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# ---------------------------
# Settings
# ---------------------------
MODEL_PATH = "./models/ppo_lunarlander_final"
NUM_EPISODES = 10
SEED_START = 1000      # episode i uses seed SEED_START + i
RENDER = True          # False = fast, no window

env = gym.make("LunarLander-v3", render_mode="human" if RENDER else None)
model = PPO.load(MODEL_PATH)


def classify(terminated, truncated, final_reward):
    """Outcome label for one episode (verify against your own runs)."""
    if truncated:
        return "timeout"
    if terminated and final_reward >= 100:
        return "landed"
    if terminated and final_reward <= -100:
        return "crashed"
    return "other"


episode_rewards = []
outcomes = []

for episode in range(NUM_EPISODES):

    obs, info = env.reset(seed=SEED_START + episode)
    total_reward = 0.0
    step = 0
    terminated = truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        print(
            f"\rEpisode: {episode + 1}/{NUM_EPISODES} | Step: {step} | "
            f"Action: {int(action)} | Total Reward: {total_reward:8.2f}",
            end="",
        )

    outcome = classify(terminated, truncated, reward)  # reward = last step's reward
    episode_rewards.append(total_reward)
    outcomes.append(outcome)

    print(
        f"\nEpisode {episode + 1} | Reward: {total_reward:.2f} | Steps: {step} | "
        f"Final-step reward: {reward:.1f} | Outcome: {outcome}"
    )
    print("-" * 80)

env.close()

# ---------------------------
# Summary
# ---------------------------
r = np.array(episode_rewards)
print("\n========== TEST RESULTS ==========")
print(f"Episodes:    {NUM_EPISODES}")
print(f"Mean reward: {r.mean():.2f}")
print(f"Std reward:  {r.std(ddof=1):.2f}")
print(f"Median:      {np.median(r):.2f}")
print(f"Min / Max:   {r.min():.2f} / {r.max():.2f}")
for label in ["landed", "crashed", "timeout", "other"]:
    print(f"{label.capitalize():8s}: {outcomes.count(label)}/{NUM_EPISODES}")
print("==================================")