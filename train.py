import numpy as np
import random
import pylab as plt
import torch
import gym
import laserhockey.hockey_env as lh
from PPO import ActivationFunctions, Initializers, BatchModes, PPO, Memory
from Agent import Agent, OpponentType, Mode
from AgentRegister import AgentRegister
from HiddenPrints import HiddenPrints


def train(training_episodes, training_length, eval_length,
          episode_max_steps, gamma, K_epochs, eps_clip, policy_depth,
          policy_width, value_depth, value_width, activation_function,
          initializer, policy_last_layer_scaler, value_last_layer_scaler,
          minimum_std, initial_std, handle_abandoned, reward_normalization,
          mini_batch_size, batch_mode, optimizer_lr, optimizer_weight_decay,
          optimizer_momentum, optimizer_epsilon, frame_skipping_length,
          advantage_normalization, input_normalization, update_timesteps,
          save_episodes, opponent_type, opponent_weak, default_timestep_loss,
          frame_skip_frequency, input_clipping_max_abs_value,
          gradient_clipping, lbda, filename="", seed=None, load_filename=None,
          print_config=True, load_info="best"):
    """Trains a new or existing agent in the hockey environment using the
    PPO algorithm.  After every [training_length] episodes of training
    the agent switches into evaluation mode for [eval_length] episodes.
    During evaluation the model is not updated.

    The agent also plays against his old checkpoints.  If the agent
    achieves a record evaluation performance that is also higher than
    the evaluations before, the current model checkpoint is added to a
    list.  From now on his training opponent is then sampled from that
    list to encourage diversity.

    Args:
        training_episodes (unsigned int):
            Number of episodes to train in total
        training_length (unsigned it):
            The agent will train for that many episodes before he
            switches into evaluation mode.  For efficiency reasons it is
            recommended to choose a higher number here than for
            eval_length.
        eval_length (unsigned int):
            After training the agent is evaluated for that many
            episodes.  He then switches to training mode again.
        episode_max_steps (unsigned int):
            Maximum number of timesteps the agant trains within an
            epsiode before it is interrupted.  230 is the default value
            of the environment.
        gamma (float:[0-1]):
            Discount factor gamma
        K_epochs (int):
            Number of times the model is updated during an update
            process (see also epochs of Stochastic Gradient Descent)
        eps_clip (float):
            Clipping parameter episolon of the PPO algorithm
        policy_depth (unsigned int):
            Number of hidden layers of policy / actor network
        policy_width (unsigned int):
            Number of neurons per hidden layer of policy / actor
            network
        value_depth (unsigned int):
            Number of hidden layers of policy / actor network
        value_width ([type]):
            Number of neurons per hidden layer of policy / actor
            network
        activation_function (ActivationFunctions):
            Activation function of policy and actor network
        initializer (Initializers):
            Initializers for the linear layers in the policy and actor
            network
        policy_last_layer_scaler (float):
            Scaler by which the last weights of the policy network are
            multiplied with after initialization
        value_last_layer_scaler (float):
            Scaler by which the last weights of the policy network are
            multiplied with after initialization
        minimum_std (float):
            Lower limit for the standard deviation used to sample
            actions.  Lower standard deviations are cut off
        initial_std (float):
            Initial standard deviation used to sample actions
        handle_abandoned (bool):
            For False the reward assigned to the last timestep of an
            episode is overwritten by the value of that last timestep /
            last state.
        reward_normalization (bool):
            If True the collected rewards during trainig time are
            normalized before updating the network. For the
            calculation of mean and std an online algorithm is used.
        mini_batch_size (unsigned int):
            Size of the mini-batch for updates of the actor/policy and
            critic/value network
        batch_mode (BatchModes):
            For SHUFFLE_RECOMPUTE_ADVANTAGES the advantages and returns
            are shuffled for every batch at the beginning of the update
            process.  For SHUFFLE this only happens for the first batch.
        optimizer_lr(float):
            Learning rate of the optimizer
        optimizer_weight_decay (float):
            Weight decay (Regularization) of the optimizer.
        optimizer_momentum (float):
            Momentum of the optimizer
        optimizer_epsilon (float):
            Epsilon of the optimizer
        frame_skipping_length (int):
            Frame skipping may lead to fast computation by using a
            calculated action for [frame_skipping_length] timesteps.
            However it might also lead to a worse performance.
        advantage_normalization (bool):
            If true the advantages of every mini-batch are normalized
            using an online mean and variance calculation
        input_normalization (bool):
            If true, the observations are normalized using an online
            mean and variance calculation
        update_timesteps (unsigned int):
            After every [update_timesteps] timesteps of training the
            model is updated.   In case that the current episode is not
            finished, the update starts after the completition of the
            current episode.  The model is also updated before switching
            to evaluation mode.
        save_episodes (unsigned int):
            After every [save_episodes] the current state of the agent
            is saved.
        opponent_type (OpponentType):
            Defines the basic opponent the agent plays against.
            The Opponent type also influences the maximum number of
            timesteps per game.
        opponent_weak (bool):
            If true the agent plays against the weak basic opponent.  In
            case that an old checkpoint version of the agent is sampled,
            this setting is not reelvant.
        default_timestep_loss (float):
            Defines a reward, whose inverted (negative) value is added
            at each timestep
        frame_skip_frequency (unsigned int):
            Every [frame_skip_frequency]th episode of training, frame
            skipping is activated with the specified length.  During the
            rest of time, the frame skipping length is set to 1.
        input_clipping_max_abs_value (float):
            Input normalization may lead to high values (e.g. at the
            beginning). Hereby normalized values lower than
            -input_clipping_max_abs_value or higher than
            +input_clipping_max_abs_value are cut off.
        gradient_clipping (int):
            Ìf not None the gradient is clipped with the here
            specified value.
        lbda (float):
            Lambda parameter of the PPO algorithm
        filename (str, optional):
            Filename under which the checkpoint is saved
            in the /checkpints folder. Defaults to "".
        seed (str, optional):
            If not None, the seed allows to replicate  performance.
            Defaults to None.
        load_filename (str, optional):
            Filename under which to load the agent if an existing agent
            is to be trained. Defaults to None. If None a new agent is
            created.
        print_config (bool, optional):
            Prints the main configuration in the console before starting
            training. Defaults to True.
        load_info (str, optional):
            Usually two checkpoints are saved of the agent: 'Best'
            refers to the model with the highest evaluation performance.
            '' refers to the agent checkpoint saved most recently.
            This allows to choose between the two when loading an
            existing agent. Defaults to "best".
    """

    max_episodes = int(np.round(
        training_episodes * (1 + eval_length / training_length), 0))

    if opponent_type == OpponentType.NORMAL:
        env = lh.HockeyEnv()
    elif opponent_type == OpponentType.DEFENDING:
        env = lh.HockeyEnv(mode=lh.HockeyEnv.TRAIN_DEFENSE)
    elif opponent_type == OpponentType.SHOOTING:
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
        agent.load(load_filename, load_info)
    else:
        agent.configure(episode_max_steps, filename, save_episodes,
                        K_epochs, eps_clip, policy_depth, policy_width,
                        value_depth, value_width, activation_function,
                        minimum_std, initial_std, policy_last_layer_scaler,
                        value_last_layer_scaler, initializer, lbda,
                        mini_batch_size, batch_mode, update_timesteps, gamma,
                        handle_abandoned, frame_skipping_length, optimizer_lr,
                        optimizer_weight_decay, optimizer_momentum,
                        optimizer_epsilon, advantage_normalization,
                        reward_normalization, input_normalization,
                        gradient_clipping, input_clipping_max_abs_value,
                        1, 0, 0, 0, 0, default_timestep_loss)

    agent.op_type = opponent_type
    agent.op_weak = opponent_weak
    agent.filename = filename
    agent.frame_skipping_activated = False
    agent.frame_skipping_length = frame_skipping_length

    if print_config:
        print(agent.config, "\n")

    current_frame_skip_frequency = frame_skip_frequency

    for i_episode in range(1, max_episodes+1):
        if opponent_type == OpponentType.DEFENDING:
            env = lh.HockeyEnv(mode=lh.HockeyEnv.TRAIN_DEFENSE)
        elif opponent_type == OpponentType.SHOOTING:
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
                agent.frame_skipping_activated = True
            elif current_frame_skip_frequency == 0:
                current_frame_skip_frequency = frame_skip_frequency
                agent.frame_skipping_activated = False
            current_frame_skip_frequency -= 1

        # Run one episode
        for _ in range(episode_max_steps):
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
