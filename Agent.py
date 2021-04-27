import numpy as np
import torch
import pickle
from enum import Enum
from PPO import PPO, Memory, ActivationFunctions, Initializers, BatchModes


class Mode(Enum):
    TRAINING = 1
    EVALUATION = 2


class OpponentType(Enum):
    DEFENDING = 1
    SHOOTING = 2
    NORMAL = 3


class Agent:
    def __init__(self, agent_register=None):
        self.mode = Mode.TRAINING
        self._current_frame_skip_pos = 0
        self._current_frame_skip_action = None
        self._current_frame_skip_reward_true = 0
        self._current_frame_skip_reward_calc = 0
        self._current_ep_timestep_counter = 0
        self._current_reward_calc = 0
        self._current_reward_true = 0
        self._current_opp_type = 1  # Defending
        self._current_opp_weak = True
        self._current_frame_skip_activated = False
        self._current_closeness_puck = 0
        self._current_touch_puck = 0
        self._current_puck_direction = 0

        self._timesteps_since_update = 0

        self._memory = Memory()
        self._memory.lengths.append(0)

        self._agent_register = agent_register

    def configure(self, environment_name, state_dim, discrete, action_dim,
                  episode_max_steps, filename, save_episodes, K_epochs,
                  eps_clip, policy_depth, policy_width, value_depth,
                  value_width, activation_function, minimum_std, initial_std,
                  policy_last_layer_scaler, value_last_layer_scaler,
                  initializer, lbda, eps_value_clip, mini_batch_size,
                  batch_mode, update_episodes, gamma, handle_abandoned,
                  frame_skipping_length, optimizer_lr, optimizer_weight_decay,
                  optimizer_momentum, optimizer_epsilon, value_normalization,
                  advantage_normalization, reward_normalization,
                  input_normalization, gradient_clipping,
                  input_clipping_max_abs_value, weighting_true_reward,
                  weighting_reward_winner, weighting_reward_closeness_puck,
                  weighting_reward_touch_puck, weighting_reward_puck_direction,
                  default_timestep_loss):

        self._config = {
            # Environment & general setting
            'environment_name': environment_name,
            'state_dim': state_dim,
            'discrete': discrete,
            'action_dim': action_dim,
            'episode_max_steps': episode_max_steps,
            'filename': filename,
            'save_episodes': save_episodes,

            # Policy losses
            'K_epochs': K_epochs,
            'eps_clip': eps_clip,

            # Network architecture
            'policy_depth': policy_depth,
            'policy_width': policy_width,
            'value_depth': value_depth,
            'value_width': value_width,
            'activation_function': activation_function,
            'minimum_std': minimum_std,
            'initial_std': initial_std,
            'policy_last_layer_scaler': policy_last_layer_scaler,
            'value_last_layer_scaler': value_last_layer_scaler,
            'initializer': initializer,

            # Advantage estimation
            'lbda': lbda,
            'eps_value_clip': eps_value_clip,

            # Training setup
            'mini_batch_size': mini_batch_size,
            'batch_mode': batch_mode,
            'update_episodes': update_episodes,

            # Time
            'gamma': gamma,
            'handle_abandoned': handle_abandoned,
            'frame_skipping_length': frame_skipping_length,

            # Optimizer
            'optimizer_lr': optimizer_lr,
            'optimizer_weight_decay': optimizer_weight_decay,
            'optimizer_momentum': optimizer_momentum,
            'optimizer_epsilon': optimizer_epsilon,

            # Normalization and clipping
            'value_normalization': value_normalization,
            'advantage_normalization': advantage_normalization,
            'reward_normalization': reward_normalization,
            'input_normalization': input_normalization,
            'gradient_clipping': gradient_clipping,
            'input_clipping_max_abs_value': input_clipping_max_abs_value,

            # weight calculation
            'weighting_true_reward': weighting_true_reward,
            'weighting_reward_winner': weighting_reward_winner,
            'weighting_reward_closeness_puck': weighting_reward_closeness_puck,
            'weighting_reward_touch_puck': weighting_reward_touch_puck,
            'weighting_reward_puck_direction': weighting_reward_puck_direction,
            'default_timestep_loss': default_timestep_loss
        }

        self._stats = {
            'global_timestep_counter': 0,
            'ep_counter': 1,
            'learning_ep_counter': 1,

            'ep_last_training': 1,
            'episode_last_switch_to_evaluation': 0,

            'episode_mode': [],
            'ep_rewards_calc': [],
            'ep_rewards_true': [],
            'ep_closeness_puck': [],
            'ep_touch_puck': [],
            'ep_puck_direction': [],
            'ep_length':  [],
            'ep_wins':  [],
            'ep_eval_results': [[], [], [], [], [], []],
            'ep_opp_weak': [],
            'ep_opp_type': [],
            'ep_frame_skip': []
        }

        self._configure_ppo()

    def _configure_ppo(self):
        self.ppo = PPO(
            self._config["state_dim"], self._config["discrete"],
            self._config["action_dim"], self._config["K_epochs"],
            self._config["eps_clip"], self._config["policy_depth"],
            self._config["policy_width"], self._config["value_depth"],
            self._config["value_width"], self._config["activation_function"],
            self._config["minimum_std"], self._config["initial_std"],
            self._config["policy_last_layer_scaler"],
            self._config["value_last_layer_scaler"], self._config["initializer"],
            self._config["lbda"], self._config["eps_value_clip"],
            self._config["mini_batch_size"], self._config["batch_mode"],
            self._config["gamma"], self._config["handle_abandoned"],
            self._config["frame_skipping_length"], self._config["optimizer_lr"],
            self._config["optimizer_weight_decay"],
            self._config["optimizer_momentum"],
            self._config["optimizer_epsilon"],
            self._config["value_normalization"],
            self._config["advantage_normalization"],
            self._config["reward_normalization"],
            self._config["input_normalization"],
            self._config["gradient_clipping"],
            self._config["input_clipping_max_abs_value"]
        )

    def act(self, state):
        if self.mode == Mode.TRAINING:
            if self._current_frame_skip_pos == 0:
                self._current_frame_skip_pos = (
                    self._config["frame_skipping_length"]
                    if self._current_frame_skip_activated else 1)
                action = self.ppo.act(state, self._memory)
                self._current_frame_skip_action = action
            else:
                action = self._current_frame_skip_action

            self._current_frame_skip_pos -= 1
        else:
            action = self.ppo.act(state)

        return action

    def feedback(self, reward, info, done, state):
        # Calculate enginered reward
        reward_calc = self._calculate_reward(reward, info, done)

        # Frame skipping: Return same action
        if (self._current_frame_skip_pos > 0 and
                not done and not self.mode == Mode.EVALUATION):
            self._current_frame_skip_reward_calc += reward
            self._current_frame_skip_reward_true += reward_calc
            return

        elif ((self._current_frame_skip_pos == 0 or done)
                and not self.mode == Mode.EVALUATION):
            reward_calc = self._current_frame_skip_reward_calc + reward_calc
            reward = self._current_frame_skip_reward_true + reward

            self._current_frame_skip_reward_calc = 0
            self._current_frame_skip_reward_true = 0

        if (self.mode == Mode.TRAINING and
                (self._current_frame_skip_pos == 0 or done)):
            self._memory.rewards.append(reward_calc)
            self._memory.is_terminals.append(done)
            self._timesteps_since_update += 1

        # Increase all temporary statistics numbers
        self._current_ep_timestep_counter += 1
        self._stats["global_timestep_counter"] += 1
        self._current_reward_calc += reward_calc
        self._current_reward_true += reward
        self._current_closeness_puck += info["reward_closeness_to_puck"]
        self._current_touch_puck += info["reward_touch_puck"]
        self._current_puck_direction += info["reward_puck_direction"]
        self._memory.lengths[-1] += 1

        if done or (self._current_ep_timestep_counter
                    % self._config["episode_max_steps"] == 0):
            # Finalize last episode
            self._stats["ep_counter"] += 1
            if self.mode == Mode.TRAINING:
                self._stats["learning_ep_counter"] += 1
            self._stats["episode_mode"].append(self.mode.value)
            self._stats["ep_rewards_calc"].append(self._current_reward_calc)
            self._stats["ep_rewards_true"].append(self._current_reward_true)
            self._stats["ep_closeness_puck"].append(
                self._current_closeness_puck)
            self._stats["ep_touch_puck"].append(self._current_touch_puck)
            self._stats["ep_puck_direction"].append(
                self._current_puck_direction)
            self._stats["ep_length"].append(self._current_ep_timestep_counter)
            self._stats["ep_wins"].append(info["winner"])
            self._stats["ep_opp_type"].append(self._current_opp_type)
            self._stats["ep_opp_weak"].append(self._current_opp_weak)

            doc_frame_skip = 1
            if (self.mode == Mode.EVALUATION and
                    self._current_frame_skip_activated):
                doc_frame_skip = self._stats["frame_skipping_length"]
            self._stats["ep_frame_skip"].append(doc_frame_skip)

            self._memory.winners.append(info["winner"])
            self._memory.final_states.append(state.reshape(-1))

            # Prepare next expisode: Reset temporary statistics
            self._current_frame_skip_pos = 0
            self._current_reward_calc = 0
            self._current_reward_true = 0
            self._current_ep_timestep_counter = 0
            self.current_closeness_to_puck = 0
            self._current_touch_puck = 0
            self._current_puck_direction = 0

            # Prepare next episode: Set next params
            self._memory.lengths.append(0)

            if self._stats["ep_counter"] % self._config["save_episodes"] == 0:
                self.save()

            if (self.mode == Mode.TRAINING and self._timesteps_since_update >=
                    self._config["update_episodes"]):
                self._update()

    def _calculate_reward(self, reward, info, done):
        if self.mode == Mode.EVALUATION:
            return reward
        elif done:
            value = (reward * self._config["weighting_true_reward"]
                     + info["winner"] * self._config["weighting_reward_winner"]
                     + (info["reward_closeness_to_puck"]
                        * self._config["weighting_reward_closeness_puck"])
                     + ((info["reward_touch_puck"]
                        * self._config["weighting_reward_touch_puck"]))
                     + (info["reward_puck_direction"]
                        * self._config["weighting_reward_puck_direction"])
                     - self._config["default_timestep_loss"])
            return value
        else:
            return 0

    def change_mode(self, print_results=False):
        if self._current_ep_timestep_counter != 0:
            raise Exception("Can't switch mode during episode")

        if self.mode == Mode.TRAINING:
            if len(self._memory.actions) > 0:
                self._update()

            self.change_frame_skipping_mode(False)
            self.mode = Mode.EVALUATION
            self._stats["episode_last_switch_to_evaluation"] = \
                self._stats["ep_counter"]

        elif self.mode == Mode.EVALUATION:
            self._compute_statistics()

            self.mode = Mode.TRAINING
            self._stats["ep_last_training"] = \
                self._stats["ep_counter"]

    def _compute_statistics(self):

        train_start = self._stats["ep_last_training"]-1
        train_end = self._stats["episode_last_switch_to_evaluation"]-1  # -2+1
        eval_start = self._stats["episode_last_switch_to_evaluation"]-1
        eval_end = len(self._stats["ep_rewards_calc"])  # -1+1=0

        train_rewards_arr = np.asarray(self._stats["ep_rewards_true"])[
            train_start:train_end]
        eval_rewards_arr = np.asarray(self._stats["ep_rewards_true"])[
            eval_start:eval_end]

        training_matches = np.asarray(self._stats["ep_wins"])[
            train_start:train_end]
        eval_matches = np.asarray(self._stats["ep_wins"])[
            eval_start:eval_end]

        avg_train_rewards = (np.sum(train_rewards_arr)
                             / len(train_rewards_arr))
        avg_eval_rewards = (np.sum(eval_rewards_arr)
                            / len(eval_rewards_arr))

        train_wins, train_lost, train_draws = self._calculateWLDRates(
            training_matches, eval_start, eval_end)

        eval_wins, eval_lost, eval_draws = self._calculateWLDRates(
            eval_matches, eval_start, eval_end)

        hist_index = 1 if self._current_opp_weak else 0
        hist_index += 2*(self._current_opp_type.value-1)

        save_note = ""
        if len(self._stats["ep_eval_results"][hist_index]) == 0:
            past_max_values = [-1000000]
        else:
            past_max_values = np.max(np.asarray(self._stats
                                                ["ep_eval_results"]
                                                [hist_index]))

        if past_max_values < avg_eval_rewards:
            save_note = " - Checkpoint saved"
            self.save("best")
            if (self._agent_register is not None
                    and len(self._stats["ep_eval_results"][hist_index]) > 0
                    and avg_eval_rewards >= 4):
                self._agent_register.add_agent(
                    self._config["filename"], "best")
                save_note = save_note + " - added"

            self._stats["ep_eval_results"][hist_index].append(
                avg_eval_rewards)

        print(("{}: ## Learn(R;W,D,L in %): {:.1f}, {:.0f}, {:.0f}, {:.0f} " +
              "Eval(R;W,D,L in %): {:.1f}, {:.0f}, {:.0f}, {:.0f} {}").format(
                  self._stats["ep_counter"]-1,
                  avg_train_rewards, train_wins, train_draws, train_lost,
                  avg_eval_rewards, eval_wins, eval_draws, eval_lost,
                  save_note))

    def _calculateWLDRates(cls, matches, start, end):
        count = end-start

        wins = np.sum(np.where(matches > 0, 1, 0))
        lost = np.sum(np.where(matches < 0, 1, 0))
        draws = np.sum(np.where(matches == 0, 1, 0))

        return wins / count * 100, lost / count * 100, draws / count * 100

    def _update(self):
        if self._memory.lengths[-1] == 0:
            del self._memory.lengths[-1]
        self.ppo.update(self._memory)
        self._memory.clear_memory()
        self._memory.lengths.append(0)
        self._timesteps_since_update = 0

    def change_frame_skipping_mode(self, activate):
        if self._current_ep_timestep_counter != 0:
            raise Exception("Can't switch mode during episode")
        if self.mode == Mode.EVALUATION:
            raise Exception("Can't be activated during evaluation")

        self._current_frame_skip_activated = activate

    def save(self, info=""):
        filename = "checkpoints/{}_{}_{}.pth".format(
            self._config["environment_name"], self._config["filename"], info)

        torch.save({
            'policy': self.ppo.policy.state_dict(),
            'policy_old': self.ppo.policy_old.state_dict(),
            'optimizer': self.ppo.optimizer.state_dict(),
            'configuration': self._config,
            'statistics': self._stats,
            'input_normalizer': pickle.dumps(self.ppo.input_normalizer),
            'advantage_normalizer':
                pickle.dumps(self.ppo.advantage_normalizer),
            'reward_normalizer': pickle.dumps(self.ppo.reward_normalizer),
            'memory': self._memory,
        }, filename)

    def load(self, environment_name, filename, info=""):
        filename = "checkpoints/{}_{}_{}.pth".format(
            environment_name, filename, info)
        checkpoint = torch.load(filename)

        self._config = checkpoint["configuration"]
        self._stats = checkpoint["statistics"]
        self._stats = checkpoint["statistics"]

        self._configure_ppo()
        self._memory = checkpoint["memory"]

        self.ppo.policy.load_state_dict(checkpoint["policy"])
        self.ppo.policy_old.load_state_dict(checkpoint["policy_old"])
        self.ppo.optimizer.load_state_dict(checkpoint["optimizer"])
        self.ppo.input_normalizer = pickle.loads(
            checkpoint["input_normalizer"])

        self.ppo.advantage_normalizer = pickle.loads(
            checkpoint["advantage_normalizer"])
        self.ppo.reward_normalizer = pickle.loads(
            checkpoint["reward_normalizer"])

    @property
    def config(self):
        return self._config

    @property
    def op_type(self):
        return self._current_opp_type

    @op_type.setter
    def op_type(self, value):
        if self._current_ep_timestep_counter != 0:
            raise Exception("Can't switch opponent_type during episode.")
        self._current_opp_type = value

    @property
    def opp_weak(self):
        return self._current_opp_weak

    @opp_weak.setter
    def opp_weak(self, value):
        if self._current_ep_timestep_counter != 0:
            raise Exception(("Can't switch opponent weakness "
                             + "during a running episode."))
        self._current_opp_weak = value

    @property
    def filename(self):
        return self._config["filename"]

    @filename.setter
    def filename(self, value):
        self._config["filename"]

    @property
    def frame_skipping_activated(self):
        return self._current_frame_skip_activated

    @frame_skipping_activated.setter
    def frame_skipping_activated(self, value):
        if self._current_ep_timestep_counter != 0:
            raise Exception(("Can't switch frame skipping skipping activation "
                             + "during a running episode."))
        self._current_frame_skip_activated = value

    @property
    def frame_skipping_length(self):
        return self._current_frame_skip_activated

    @frame_skipping_length.setter
    def frame_skipping_length(self, value):
        if self._current_ep_timestep_counter != 0:
            raise Exception(("Can't switch frame skipping length during "
                             + "a running episode."))
        self._config["frame_skipping_length"] = value
