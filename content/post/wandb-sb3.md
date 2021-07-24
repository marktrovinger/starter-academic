---
date: 2021-07-23
title: Tracking Experiments, Now Easier!
summary: Using Weights and Bias's new SB3 integration makes experiment tracking easier.
tags: ["reinforcement learning","experiment","wandb", "sb3"]
---

One aspect of doing reinforcement learning research that has been more annoying than I would like is keeping track of experiments and experimental results. While this may sound a bit like an infomercial, I have to say that using Weights and Bias's integration with Stable Baselines 3 has made experiment tracking way easier than it was before!

Using it is pretty simple, update to the latest version of WandB and then use:

```
from wandb.integration.sb3 import WandbCallback

model.learn(..., callback=WandbCallback())
```

The basic example from WandB uses Tensorboard's output to log metrics:

```
import time
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback
import wandb

config = {"policy_type": "MlpPolicy", "total_timesteps": 25000}
experiment_name = f"PPO_{int(time.time())}"

# Initialise a W&B run
wandb.init(
    name=experiment_name,
    project="sb3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

def make_env():
    env = gym.make("CartPole-v1")
    env = Monitor(env)  # record stats such as returns
    return env

env = DummyVecEnv([make_env])

env = VecVideoRecorder(env, "videos",
    record_video_trigger=lambda x: x % 2000 == 0, video_length=200)

model = PPO(config["policy_type"], env, verbose=1,
    tensorboard_log=f"runs/{experiment_name}")

# Add the WandbCallback 
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_freq=1000,
        model_save_path=f"models/{experiment_name}",
    ),
)
```