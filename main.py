from train import train
from PPO import ActivationFunctions, Initializers, BatchModes
from Agent import OpponentType


def main():
    #import os
    #os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    DEFAULT_policy_depth = 2
    DEFAULT_policy_width = 128
    DEFAULT_value_depth = 4
    DEFAULT_value_width = 256
    DEFAULT_activation_function = ActivationFunctions.TANH  # TESTING
    DEFAULT_initializer = Initializers.ORTHOGONAL
    DEFAULT_value_last_layer_scaler = 0.001
    DEFAULT_policy_last_layer_scaler = 0.01
    DEFAULT_minimum_std = 0.01
    DEFAULT_initial_std = 0.5

    DEFAULT_input_clipping_max_abs_value = 10
    DEFAULT_gradient_clipping = 10
    DEFAULT_reward_normalizations = False
    DEFAULT_advantage_normalization = False
    DEFAULT_input_normalization = True

    DEFAULT_gamma = 0.95
    DEFAULT_handle_abandoned = True
    DEFAULT_frame_skipping_length = 1
    DEFAULT_frame_skipping_frequency = None

    DEFAULT_lbda = 0.95

    DEFAULT_eps_clip = 0.25
    DEFAULT_K_epochs = 10
    DEFAULT_mini_batch_size = 128
    DEFAULT_batch_mode = BatchModes.SHUFFLE_RECOMPUTE_ADVANTAGES
    DEFAULT_update_timesteps = 20000

    DEFAULT_optimizer_lrs = 0.0003
    DEFAULT_optimizer_weight_decay = 0.0
    DEFAULT_optimizer_momentum = 0.9
    DEFAULT_optimizer_epsilon = 1e-8

    DEFAULT_training_length = 10
    DEFAULT_eval_length = 4
    DEFAULT_episode_max_steps = 402
    DEFAULT_save_episode = DEFAULT_training_length + DEFAULT_eval_length

    DEFAULT_opponent_type = OpponentType.NORMAL
    DEFAULT_opponent_weak = False
    DEFAULT_timestep_loss = 0

    DEFAULT_max_training_episodes = 40000

    seed_1 = 123456
    title = save_filename = "Train report settings"

    # train(DEFAULT_max_training_episodes, DEFAULT_training_length, DEFAULT_eval_length, DEFAULT_episode_max_steps,
    #      DEFAULT_gamma, DEFAULT_K_epochs, DEFAULT_eps_clip, DEFAULT_policy_depth, DEFAULT_policy_width, DEFAULT_value_depth, DEFAULT_value_width, DEFAULT_activation_function,
    #      DEFAULT_initializer, DEFAULT_policy_last_layer_scaler, DEFAULT_value_last_layer_scaler, DEFAULT_minimum_std,
    #      DEFAULT_initial_std, DEFAULT_handle_abandoned, DEFAULT_reward_normalizations, DEFAULT_mini_batch_size, DEFAULT_batch_mode, DEFAULT_optimizer_lrs,
    #      DEFAULT_optimizer_weight_decay, DEFAULT_optimizer_momentum, DEFAULT_optimizer_epsilon, DEFAULT_frame_skipping_length, DEFAULT_advantage_normalization,
    #      DEFAULT_input_normalization, DEFAULT_update_timesteps, DEFAULT_save_episode, DEFAULT_opponent_type,
    #      DEFAULT_opponent_weak, DEFAULT_timestep_loss, DEFAULT_frame_skipping_frequency,
    #      DEFAULT_input_clipping_max_abs_value, DEFAULT_gradient_clipping, DEFAULT_lbda, save_filename, seed_1, print_config=True, load_info="")
    train(DEFAULT_max_training_episodes, DEFAULT_training_length, DEFAULT_eval_length, DEFAULT_save_episode,
          filename=save_filename, seed=seed_1, print_config=True, load_info="")


if __name__ == "__main__":
    main()
