# RL Action Failures

Experiments with a PPO agent in Gymnasium's `LunarLander-v3` environment, focused on how a trained policy behaves when one or more commanded actions silently fail at execution time.

## What this repository contains

- `train.py` trains a PPO (`MlpPolicy`) agent for 500,000 steps and saves checkpoints every 50,000 steps.
- `failure_wrapper.py` provides `ActionFailureWrapper`, which maps specified failed discrete actions to action `0` (*do nothing*) without informing the agent.
- `test.py` runs a rendered baseline evaluation of the saved model.
- `test_failures.py` runs the same style of evaluation with configured action failures and reports commanded versus executed actions.
- `evaluate.py` evaluates 100 deterministic, seeded baseline episodes and writes a per-episode CSV to `results/`.
- `models/`, `logs/`, and `results/` contain the included trained-model artifacts, TensorBoard logs, and baseline output.

For LunarLander, the discrete action mapping is:

| Action | Meaning |
| --- | --- |
| `0` | Do nothing |
| `1` | Fire left orientation engine |
| `2` | Fire main engine |
| `3` | Fire right orientation engine |

## Setup

Use Python 3 and install the experiment dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install "gymnasium[box2d]" stable-baselines3 numpy pandas
```

If Box2D installation fails on your platform, follow the [Gymnasium Box2D installation guidance](https://gymnasium.farama.org/environments/box2d/).

## Train

Train with the default seed (`42`):

```bash
python train.py
```

Or provide a different seed:

```bash
python train.py 7
```

Training writes seed-specific checkpoints to `models/seed<seed>/` and TensorBoard data to `logs/seed<seed>/`.

## Evaluate the baseline

The included scripts load `models/ppo_lunarlander_final`.

```bash
# Fast, deterministic 100-episode evaluation; writes results/baseline_normal.csv
python evaluate.py

# Ten-episode rendered evaluation; set RENDER = False in test.py for headless runs
python test.py
```

## Simulate an action failure

`test_failures.py` defaults to failing action `3` (the right orientation engine):

```python
FAILED_ACTIONS = (3,)
```

Run it with:

```bash
python test_failures.py
```

To test another condition, edit `FAILED_ACTIONS`. Examples include `()` for the no-failure control and `(1, 3)` for simultaneous left- and right-engine failures. A failed command is executed as action `0`, while the wrapper records both values in `info` for analysis.

## Reproducibility notes

- Training uses PPO with a fixed default seed of `42`.
- Evaluation uses deterministic model predictions and explicit episode seeds.
- `test.py` and `test_failures.py` use the same seed sequence by default, enabling matched baseline/failure scenarios.
