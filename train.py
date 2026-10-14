import sys
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# ---------------------------
# Settings
# ---------------------------
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
TOTAL_TIMESTEPS = 500_000

MODEL_DIR = f"./models/seed{SEED}/"
LOG_DIR = f"./logs/seed{SEED}/"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Keep identical for every method you compare (defaults for now)
PPO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.0,
)

# ---------------------------
# Environments
# ---------------------------
env = gym.make("LunarLander-v3")
eval_env = Monitor(gym.make("LunarLander-v3"))  # separate env, never trained on

# ---------------------------
# Agent
# ---------------------------
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    seed=SEED,
    **PPO_KWARGS,
)

# ---------------------------
# Callbacks
# ---------------------------
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path=MODEL_DIR,
    name_prefix="ppo_lunarlander",
)

eval_callback = EvalCallback(
    eval_env,
    eval_freq=10_000,
    n_eval_episodes=20,
    deterministic=True,
    log_path=LOG_DIR,        # saves evaluations.npz (learning curve data)
    best_model_save_path=None,  # don't pick "best" models (avoids selection bias)
)

# ---------------------------
# Train and save
# ---------------------------
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[checkpoint_callback, eval_callback],
)

model.save(f"{MODEL_DIR}ppo_lunarlander_final")
env.close()
eval_env.close()