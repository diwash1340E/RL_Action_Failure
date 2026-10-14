import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from failure_wrapper import ActionFailureWrapper

print("Starting failure test...", flush=True)

# ---------------------------
# Settings
# ---------------------------
MODEL_PATH = "./models/ppo_lunarlander_final"
NUM_EPISODES = 10
SEED_START = 1000            # same seeds as test.py -> matched scenarios
FAILED_ACTIONS = (3,)        # () = no failure, (3,) = right thruster
RENDER = False               # set True to watch (slower)

env = gym.make("LunarLander-v3", render_mode="human" if RENDER else None)
env = ActionFailureWrapper(env, failed_actions=FAILED_ACTIONS)
model = PPO.load(MODEL_PATH)
print("Model and environment loaded.", flush=True)


def classify(terminated, truncated, final_reward):
    if truncated:
        return "timeout"
    if terminated and final_reward >= 100:
        return "landed"
    if terminated and final_reward <= -100:
        return "crashed"
    return "other"


episode_rewards, outcomes = [], []

for episode in range(NUM_EPISODES):
    obs, info = env.reset(seed=SEED_START + episode)   # wrapper resets its counter
    total_reward, steps = 0.0, 0
    terminated = truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        print(
            f"\rEpisode: {episode + 1}/{NUM_EPISODES} | Step: {steps} | "
            f"Cmd: {info['commanded_action']} -> Exec: {info['executed_action']} | "
            f"Total: {total_reward:8.2f}",
            end="",
            flush=True,
        )

    outcome = classify(terminated, truncated, reward)
    episode_rewards.append(total_reward)
    outcomes.append(outcome)

    rate = 100 * env.failed_action_attempts / steps
    print(
        f"\nEpisode {episode + 1} | {outcome} | Reward: {total_reward:.2f} | "
        f"Steps: {steps} | Failed-action attempts: "
        f"{env.failed_action_attempts} ({rate:.1f}% of steps)",
        flush=True,
    )
    print("-" * 90, flush=True)

env.close()

r = np.array(episode_rewards)
print("\n========== FAILURE TEST RESULTS ==========")
print(f"Failed actions: {FAILED_ACTIONS}")
print(f"Episodes:    {NUM_EPISODES}")
print(f"Mean reward: {r.mean():.2f}   Std: {r.std(ddof=1):.2f}   Median: {np.median(r):.2f}")
print(f"Min / Max:   {r.min():.2f} / {r.max():.2f}")
for label in ["landed", "crashed", "timeout", "other"]:
    print(f"{label.capitalize():8s}: {outcomes.count(label)}/{NUM_EPISODES}")
print("==========================================", flush=True)