import torch
from PPO import ActivationFunctions, Initializers, BatchModes, PPO, Memory
from Agent import Agent, OpponentType, Mode
import gym
import numpy as np
import random
import laserhockey.hockey_env as lh
import sys
import os
import pylab as plt
from AgentRegister import AgentRegister


def train_register_change(training_episodes, training_length, eval_length,
                          max_timesteps_per_episode, gamma, K_epochs, eps_clip,
                          eps_value_clip, policy_depth, policy_width,
                          value_depth, value_width, activation_function,
                          initializer, policy_last_layer_scaler,
                          value_last_layer_scaler, minimum_std, initial_std,
                          handle_abandoned, reward_normalization,
                          mini_batch_size, batch_mode, optimizer_lr,
                          optimizer_weight_decay, optimizer_momentum,
                          optimizer_epsilon,
                          frame_skipping_length, value_normalization,
                          advantage_normalization, input_normalization,
                          update_episodes, save_episodes, opponent_type,
                          opponent_weak, default_timestep_loss,
                          frame_skip_frequency, input_clipping_max_abs_value,
                          gradient_clipping, lbda, filename="", seed=None,
                          load_filename=None, print_config=True,
                          load_info="best"):

    environment_name = 'Hockey-v0'
    state_dim = 18
    discrete = False
    action_dim = 4
    max_episodes = int(np.round(
        training_episodes * (1 + eval_length / training_length), 0))

    if opponent_type == OpponentType.Normal:
        env = lh.HockeyEnv()
    elif opponent_type == OpponentType.Defending:
        env = lh.HockeyEnv(mode=lh.HockeyEnv.TRAIN_DEFENSE)
    elif opponent_type == OpponentType.Shooting:
        env = lh.HockeyEnv(mode=lh.HockeyEnv.TRAIN_SHOOTING)

    if seed is not None:
        torch.manual_seed(seed)
        env.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    ar = AgentRegister(opponent_weak, opponent_type)
    agent = Agent(ar)

    if load_filename is not None:
        print("Loaded")
        agent.load(environment_name, load_filename, load_info)
    else:
        agent.configure(environment_name, state_dim, discrete, action_dim,
                        max_timesteps_per_episode, filename, save_episodes,
                        K_epochs, eps_clip, policy_depth, policy_width,
                        value_depth, value_width, activation_function,
                        minimum_std, initial_std, policy_last_layer_scaler,
                        value_last_layer_scaler, initializer, lbda,
                        eps_value_clip, mini_batch_size, batch_mode,
                        update_episodes, gamma, handle_abandoned,
                        frame_skipping_length, optimizer_lr,
                        optimizer_weight_decay, optimizer_momentum,
                        optimizer_epsilon, value_normalization,
                        advantage_normalization, reward_normalization,
                        input_normalization, gradient_clipping,
                        input_clipping_max_abs_value)

    agent.set_opponent(opponent_type, opponent_weak)
    agent.set_filename(filename)
    agent.set_frame_skipping(frame_skipping_length, False)
    agent.set_reward_weights(1, 0, 0, 0, 0, default_timestep_loss)

    if print_config:
        print(agent.configuration, "\n")

    current_frame_skip_frequency = frame_skip_frequency

    for i_episode in range(1, max_episodes+1):
        if opponent_type == OpponentType.Defending:
            env = lh.HockeyEnv(mode=lh.HockeyEnv.TRAIN_DEFENSE)
        elif opponent_type == OpponentType.Shooting:
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


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def main():
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    from main import train_register_change
    from PPO import ActivationFunctions, Initializers, BatchModes
    from Agent import OpponentType

    DEFAULT_policy_depth = 2
    DEFAULT_policy_width = 128
    DEFAULT_value_depth = 4
    DEFAULT_value_width = 256
    DEFAULT_activation_function = ActivationFunctions.Tanh  # TESTING
    DEFAULT_initializer = Initializers.Orthogonal
    DEFAULT_value_last_layer_scaler = 0.001
    DEFAULT_policy_last_layer_scaler = 0.01
    DEFAULT_minimum_std = 0.01
    DEFAULT_initial_std = 0.5

    DEFAULT_input_clipping_max_abs_value = 10
    DEFAULT_gradient_clipping = 10
    DEFAULT_reward_normalizations = False
    DEFAULT_value_normalization = False
    DEFAULT_advantage_normalization = False
    DEFAULT_input_normalization = True

    DEFAULT_gamma = 0.95
    DEFAULT_handle_abandoned = True
    DEFAULT_frame_skipping = 1
    DEFAULT_frame_skip_interval = None

    DEFAULT_lbda = 0.95
    DEFAULT_eps_value_clip = None

    DEFAULT_eps_clip = 0.25
    DEFAULT_K_epochs = 10
    DEFAULT_mini_batch_size = 128
    DEFAULT_batch_mode = BatchModes.Shuffle_Transitions_Recompute_Advantages
    DEFAULT_update_episodes = 20000

    DEFAULT_optimizer_lrs = 0.0003
    DEFAULT_optimizer_weight_decay = 0.0
    DEFAULT_optimizer_momentum = 0.9
    DEFAULT_optimizer_epsilon = 1e-8

    DEFAULT_training_length = 10
    DEFAULT_eval_length = 4
    DEFAULT_max_timesteps_per_episode = 402
    DEFAULT_save_episode = DEFAULT_training_length + DEFAULT_eval_length

    DEFAULT_opponent_type = OpponentType.Normal
    DEFAULT_opponent_weak = False
    DEFAULT_timestep_loss = 0

    DEFAULT_max_training_episodes = 40000

    seed_1 = 123456
    title = save_filename = "Train report settings"

    train_register_change(DEFAULT_max_training_episodes, DEFAULT_training_length, DEFAULT_eval_length, DEFAULT_max_timesteps_per_episode,
                          DEFAULT_gamma, DEFAULT_K_epochs, DEFAULT_eps_clip, DEFAULT_eps_value_clip, DEFAULT_policy_depth, DEFAULT_policy_width, DEFAULT_value_depth, DEFAULT_value_width, DEFAULT_activation_function,
                          DEFAULT_initializer, DEFAULT_policy_last_layer_scaler, DEFAULT_value_last_layer_scaler, DEFAULT_minimum_std,
                          DEFAULT_initial_std, DEFAULT_handle_abandoned, DEFAULT_reward_normalizations, DEFAULT_mini_batch_size, DEFAULT_batch_mode, DEFAULT_optimizer_lrs,
                          DEFAULT_optimizer_weight_decay, DEFAULT_optimizer_momentum, DEFAULT_optimizer_epsilon, DEFAULT_frame_skipping, DEFAULT_value_normalization, DEFAULT_advantage_normalization,
                          DEFAULT_input_normalization, DEFAULT_update_episodes, DEFAULT_save_episode, DEFAULT_opponent_type,
                          DEFAULT_opponent_weak, DEFAULT_timestep_loss, DEFAULT_frame_skip_interval,
                          DEFAULT_input_clipping_max_abs_value, DEFAULT_gradient_clipping, DEFAULT_lbda, save_filename, seed_1, print_config=False, load_info="")


if __name__ == "__main__":
    main()
