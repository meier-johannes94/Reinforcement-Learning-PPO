import torch
from PPO import Activation_Functions, Initializers, Advantage_Calculation, Batch_Mode, PPO, Memory
from Agent import Agent, Opponent_Type, Mode
import gym
import numpy as np
import random
import laserhockey.hockey_env as lh
import sys
import os
import pylab as plt


def train_register_change(training_episodes, training_length, eval_length, max_timesteps_per_episode,
                          gamma, K_epochs, eps_clip, eps_value_clip, policy_depth, policy_width, value_depth, value_width, activation_function,
                          initializer, policy_last_layer_scaler,  value_last_layer_scaler, minimum_std,
                          initial_std, handle_abandoned, reward_normalization, advantage_calculation, mini_batch_size, batch_mode, optimizer_lr,
                          optimizer_weight_decay, optimizer_momentum, optimizer_epsilon, frame_skipping_length, value_normalization, advantage_normalization,
                          input_normalization, update_episodes, save_episodes, opponent_type, opponent_weak, default_timestep_loss, frame_skip_frequency,
                          input_clipping_max_abs_value, gradient_clipping, lbda,
                          filename="", seed=None, load_filename=None, print_config=True, load_info="best"):

    environment_name = 'Hockey-v0'
    state_dim = 18
    discrete = False
    action_dim = 4
    max_episodes = int(np.round(training_episodes *
                       (1 + eval_length / training_length), 0))

    if opponent_type == Opponent_Type.Normal:
        env = lh.HockeyEnv()
    elif opponent_type == Opponent_Type.Defending:
        env = lh.HockeyEnv(mode=lh.HockeyEnv.TRAIN_DEFENSE)
    elif opponent_type == Opponent_Type.Shooting:
        env = lh.HockeyEnv(mode=lh.HockeyEnv.TRAIN_SHOOTING)

    if seed is not None:
        torch.manual_seed(seed)
        env.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    ar = Agent_Register(opponent_weak, opponent_type)
    agent = Agent(ar)

    if load_filename is not None:
        agent.load(environment_name, load_filename, load_info)
    else:
        agent.configure(environment_name, state_dim, discrete, action_dim, max_timesteps_per_episode, filename, save_episodes,
                        K_epochs, eps_clip,
                        policy_depth, policy_width, value_depth, value_width, activation_function, minimum_std, initial_std,
                        policy_last_layer_scaler, value_last_layer_scaler, initializer,
                        advantage_calculation, lbda, eps_value_clip,
                        mini_batch_size, batch_mode, update_episodes,
                        gamma, handle_abandoned, frame_skipping_length,
                        optimizer_lr, optimizer_weight_decay, optimizer_momentum, optimizer_epsilon,
                        value_normalization, advantage_normalization, reward_normalization, input_normalization, gradient_clipping,
                        input_clipping_max_abs_value)

    agent.set_opponent(opponent_type, opponent_weak)
    agent.set_filename(filename)
    agent.set_frame_skipping(frame_skipping_length, False)
    agent.set_reward_weights(1, 0, 0, 0, 0, default_timestep_loss)

    if print_config:
        print(agent.configuration, "\n")

    current_frame_skip_frequency = frame_skip_frequency

    for i_episode in range(1, max_episodes+1):
        if opponent_type == Opponent_Type.Defending:
            env = lh.HockeyEnv(mode=lh.HockeyEnv.TRAIN_DEFENSE)
        elif opponent_type == Opponent_Type.Shooting:
            env = lh.HockeyEnv(mode=lh.HockeyEnv.TRAIN_SHOOTING)

        # Sample opponent
        while True:
            player2 = ar.sample_agent(agent.mode)
            if player2 is not None:
                break

        state = env.reset()
        state2 = env.obs_agent_two()

        # Activate / Deactivate frame skipping
        if frame_skip_frequency is not None and agent.mode == Mode.Training:
            if current_frame_skip_frequency == 1:
                current_frame_skip_frequency = frame_skip_frequency
                agent.change_frame_skipping_mode(True)
            if current_frame_skip_frequency == 0:
                agent.change_frame_skipping_mode(False)
            current_frame_skip_frequency -= 1

        # Run one episode
        for _ in range(max_timesteps_per_episode):
            a1 = agent.act(state)
            a2 = player2.act(state2)
            with HiddenPrints():
                state, reward, done, info = env.step(np.hstack((a1, a2)))
            agent.feedback(reward, info, done, state)
            state2 = env.obs_agent_two()

            if done:
                break

        if (i_episode) % (training_length + eval_length) == training_length:
            agent.change_mode(False)  # Switch to evaluation
        elif (i_episode) % (training_length + eval_length) == 0:
            agent.change_mode(True)  # Switch to training


class Agent_Register():
    def __init__(self, opponent_weak, opponent_mode):
        self.env_name = 'Hockey-v0'

        self.opponent_weak = opponent_weak
        self.opponent_mode = opponent_mode

        self.agents = [lh.BasicOpponent(opponent_weak)]
        self.scores = []

    def add_agent(self, filename, info):
        if len(self.agents) > 20:
            min_index = np.argmin(self.scores)
            del self.scores[min_index]
            del self.agents[min_index+1]

        new_agent = Agent()
        new_agent.load(self.env_name, filename, info)
        new_agent.change_mode(False)

        calculate_history_index = 1 if self.opponent_weak else 0
        calculate_history_index += 2*(self.opponent_mode.value-1)

        self.scores.append(
            new_agent.statistics["episode_eval_results"][calculate_history_index][-1])
        self.agents.append(new_agent)

    def sample_agent(self, mode):
        if mode == Mode.Evaluation or len(self.agents) == 1:
            return self.agents[0]
        elif mode == Mode.Training:
            scores = np.asarray(self.scores)
            scores -= np.min(scores) - 1
            logs = np.log(scores)

            prob_basic_agent = 0.8
            if len(self.agents) >= 3:
                prob_basic_agent = 0.6
            elif len(self.agents) >= 6:
                prob_basic_agent = 0.30
            elif len(self.agents) >= 10:
                prob_basic_agent = 0.20
            elif len(self.agents) >= 15:
                prob_basic_agent = 0.10

            p = (1-prob_basic_agent) * self.softmax(logs)
            p = np.insert(p, 0, prob_basic_agent)
            p /= p.sum()
            agent_index = np.random.choice(np.arange(0, len(self.agents)), p=p)
            return self.agents[agent_index]

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
