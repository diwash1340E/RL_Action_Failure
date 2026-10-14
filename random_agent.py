import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")

observation, info = env.reset()

print("Observation:", observation)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

for step in range(1000):
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()