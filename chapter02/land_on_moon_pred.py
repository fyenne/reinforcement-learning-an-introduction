# Author: Till Zemann
# License: MIT License

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

import gymnasium as gym
# environment hyperparams
from a2c_model import A2C

if __name__ == '__main__':
    n_envs = 10
    n_updates = 400
    n_steps_per_update = 128
    randomize_domain = False
    actor_weights_path = "./weights/actor_weights.h5"
    critic_weights_path = "./weights/critic_weights.h5"
    envs = gym.vector.AsyncVectorEnv(
            [
                lambda: gym.make(
                    "LunarLander-v2",
                    gravity=np.clip(
                        np.random.normal(loc=-10.0, scale=1.0), a_min=-11.99, a_max=-0.01
                    ),
                    enable_wind=np.random.choice([True, False]),
                    wind_power=np.clip(
                        np.random.normal(loc=15.0, scale=1.0), a_min=0.01, a_max=19.99
                    ),
                    turbulence_power=np.clip(
                        np.random.normal(loc=1.5, scale=0.5), a_min=0.01, a_max=1.99
                    ),
                    max_episode_steps=600,
                    render_mode="rgb_array", 
                )
                for i in range(n_envs)
            ]
        )

    # agent hyperparams
    gamma = 0.999
    lam = 0.95  # hyperparameter for GAE
    ent_coef = 0.01  # coefficient for the entropy bonus (to encourage exploration)
    actor_lr = 0.001
    critic_lr = 0.005
    obs_shape = envs.single_observation_space.shape[0]
    action_shape = envs.single_action_space.n

    # set the device
    use_cuda = False
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # init the agent

    agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, n_envs=n_envs)

    agent.actor.load_state_dict(torch.load(actor_weights_path))
    agent.critic.load_state_dict(torch.load(critic_weights_path))
    agent.actor.eval()
    agent.critic.eval()
    
    """ play a couple of showcase episodes """

    n_showcase_episodes = 3
    
    for episode in range(n_showcase_episodes):
        print(f"starting episode {episode}...")

        # create a new sample environment to get new random parameters
        if randomize_domain:
            env = gym.make(
                "LunarLander-v2",
                render_mode="human",
                gravity=np.clip(
                    np.random.normal(loc=-10.0, scale=2.0), a_min=-11.99, a_max=-0.01
                ),
                enable_wind=np.random.choice([True, False]),
                wind_power=np.clip(
                    np.random.normal(loc=15.0, scale=2.0), a_min=0.01, a_max=19.99
                ),
                turbulence_power=np.clip(
                    np.random.normal(loc=1.5, scale=1.0), a_min=0.01, a_max=1.99
                ),
                max_episode_steps=500,
            )
        else:
            env = gym.make("LunarLander-v2", render_mode="human", max_episode_steps=500)

        # get an initial state
        state, info = env.reset()

        # play one episode
        done = False
        while not done:
            # select an action A_{t} using S_{t} as input for the agent
            with torch.no_grad():
                action, _, _, _ = agent.select_action(state[None, :])

            # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
            state, reward, terminated, truncated, info = env.step(action.item())

            # update if the environment is done
            done = terminated or truncated

    env.close()