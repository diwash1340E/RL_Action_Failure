import gymnasium as gym
import numpy as np


class ActionFailureWrapper(gym.ActionWrapper):
    """
    Simulates complete failure of one or more discrete actions.

    LunarLander actions:
        0 = do nothing, 1 = left engine, 2 = main engine, 3 = right engine

    A failed action is executed as action 0 (do nothing).
    The agent is NOT told that anything failed.

    failed_actions=()   -> no failure (identity wrapper, used as the control)
    failed_actions=(3,) -> right thruster failure
    """

    def __init__(self, env, failed_actions=(3,)):
        super().__init__(env)
        self.failed_actions = set(int(a) for a in failed_actions)
        self.n_actions = env.action_space.n

        self._commanded = 0
        self._executed = 0
        self.failed_action_attempts = 0   # per episode (reset in reset())

    @property
    def capability(self):
        """Binary capability vector, e.g. [1, 1, 1, 0]."""
        return np.array(
            [0 if a in self.failed_actions else 1 for a in range(self.n_actions)],
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        self.failed_action_attempts = 0
        return self.env.reset(**kwargs)

    def action(self, action):
        self._commanded = int(action)
        if self._commanded in self.failed_actions:
            self.failed_action_attempts += 1
            self._executed = 0            # failed: do nothing
        else:
            self._executed = self._commanded
        return self._executed

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["commanded_action"] = self._commanded
        info["executed_action"] = self._executed
        return obs, reward, terminated, truncated, info