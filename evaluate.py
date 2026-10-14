import os
import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# ============================================================
# SETTINGS
# ============================================================
MODEL_PATH = "./models/ppo_lunarlander_final"
NUM_EPISODES = 100
EVAL_SEED_START = 10_000        # far from the training seed (42)
RESULTS_DIR = "./results"


# ============================================================
# REUSABLE FUNCTIONS
# ============================================================
def classify(terminated, truncated, final_reward):
    if truncated:
        return "timeout"
    if terminated and final_reward >= 100:
        return "landed"
    if terminated and final_reward <= -100:
        return "crashed"
    return "other"


def evaluate_policy(model, env, condition, num_episodes, seed_start):
    """Run num_episodes with matched seeds; return one row per episode."""
    rows = []
    for i in range(num_episodes):
        seed = seed_start + i
        obs, info = env.reset(seed=seed)

        total_reward, steps, final_reward = 0.0, 0, 0.0
        action_counts = np.zeros(4, dtype=int)   # commanded actions
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action_counts[int(action)] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            final_reward = reward
            steps += 1

        rows.append({
            "condition": condition,
            "episode": i,
            "seed": seed,
            "total_reward": total_reward,
            "steps": steps,
            "final_reward": final_reward,
            "outcome": classify(terminated, truncated, final_reward),
            "cmd_action_0": action_counts[0],
            "cmd_action_1": action_counts[1],
            "cmd_action_2": action_counts[2],
            "cmd_action_3": action_counts[3],
        })
    return pd.DataFrame(rows)


def wilson_interval(k, n, z=1.96):
    """95% Wilson confidence interval for a proportion, in percent."""
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return 100 * (center - half), 100 * (center + half)


def summarize(df, title):
    n = len(df)
    r = df["total_reward"]
    print("\n" + "=" * 55)
    print(f"  {title}")
    print("=" * 55)
    print(f"Episodes:     {n}")
    print(f"Mean reward:  {r.mean():.2f}  (std {r.std():.2f}, median {r.median():.2f})")
    print(f"Min / Max:    {r.min():.2f} / {r.max():.2f}")
    print(f"Mean length:  {df['steps'].mean():.1f} steps")
    for label in ["landed", "crashed", "timeout", "other"]:
        k = (df["outcome"] == label).sum()
        lo, hi = wilson_interval(k, n)
        print(f"{label.capitalize():8s}: {k:3d} ({100*k/n:.1f}%, 95% CI {lo:.1f}-{hi:.1f})")
    total_steps = df["steps"].sum()
    print(f"Action 3 share of commanded actions: "
          f"{100 * df['cmd_action_3'].sum() / total_steps:.1f}%")


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model = PPO.load(MODEL_PATH)

    env = gym.make("LunarLander-v3")
    df = evaluate_policy(model, env, "normal", NUM_EPISODES, EVAL_SEED_START)
    env.close()

    out_path = os.path.join(RESULTS_DIR, "baseline_normal.csv")
    df.to_csv(out_path, index=False)

    summarize(df, "NORMAL PPO BASELINE")
    print(f"\nSaved to: {out_path}")