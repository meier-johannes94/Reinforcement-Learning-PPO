import numpy as np
import torch
import pickle
from enum import Enum
from PPO import PPO, Memory, ActivationFunctions, Initializers, BatchModes


class Mode(Enum):
    Training = 1
    Evaluation = 2


class OpponentType(Enum):
    Defending = 1
    Shooting = 2
    Normal = 3


class Agent:
    def __init__(self, agent_register=None):
        self.mode = Mode.Training
        self.current_frame_skip_pos = 0
        self.current_frame_skip_action = None
        self.current_frame_skip_reward_true = 0
        self.current_frame_skip_reward_calc = 0
        self.current_ep_timestep_counter = 0
        self.current_reward_calc = 0
        self.current_reward_true = 0
        self.current_opp_type = 1  # Defending
        self.current_opp_weak = True
        self.current_frame_skip_activated = False
        self.current_closeness_puck = 0
        self.current_touch_puck = 0
        self.current_puck_direction = 0

        self.timesteps_since_update = 0

        self.memory = Memory()
        self.memory.lengths.append(0)

        self.agent_register = agent_register

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
                  input_clipping_max_abs_value):

        self.config = {
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
            'weighting_true_reward': 1,
            'weighting_reward_winner': 0,
            'weighting_reward_closeness_puck': 0,
            'weighting_reward_touch_puck': 0,
            'weighting_reward_puck_direction': 0,
            'default_timestep_loss': 0
        }

        self.stats = {
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

        print(self.config)
        self.configure_PPO()

    def configure_PPO(self):
        self.ppo = PPO(
            self.config["state_dim"], self.config["discrete"],
            self.config["action_dim"], self.config["K_epochs"],
            self.config["eps_clip"], self.config["policy_depth"],
            self.config["policy_width"], self.config["value_depth"],
            self.config["value_width"], self.config["activation_function"],
            self.config["minimum_std"], self.config["initial_std"],
            self.config["policy_last_layer_scaler"],
            self.config["value_last_layer_scaler"], self.config["initializer"],
            self.config["lbda"], self.config["eps_value_clip"],
            self.config["mini_batch_size"], self.config["batch_mode"],
            self.config["gamma"], self.config["handle_abandoned"],
            self.config["frame_skipping_length"], self.config["optimizer_lr"],
            self.config["optimizer_weight_decay"],
            self.config["optimizer_momentum"],
            self.config["optimizer_epsilon"],
            self.config["value_normalization"],
            self.config["advantage_normalization"],
            self.config["reward_normalization"],
            self.config["input_normalization"],
            self.config["gradient_clipping"],
            self.config["input_clipping_max_abs_value"])

    def change_mode(self, print_results=False):
        if self.current_ep_timestep_counter != 0:
            raise Exception("Can't switch mode during episode")

        if self.mode == Mode.Training:
            if len(self.memory.actions) > 0:
                self.update()

            self.change_frame_skipping_mode(False)
            self.mode = Mode.Evaluation
            self.stats["episode_last_switch_to_evaluation"] = \
                self.stats["ep_counter"]

        elif self.mode == Mode.Evaluation:
            self.cycle_statistics()

            self.mode = Mode.Training
            self.stats["ep_last_training"] = \
                self.stats["ep_counter"]

    def _calculateWLDRates(cls, matches, start, end):
        count = end-start

        wins = np.sum(np.where(matches > 0, 1, 0))
        lost = np.sum(np.where(matches < 0, 1, 0))
        draws = np.sum(np.where(matches == 0, 1, 0))

        return wins / count * 100, lost / count * 100, draws / count * 100

    # Calculates training & evaluation performance, prints
    # and recoreds performs, adds agent to AgentRegistry

    def cycle_statistics(self):

        train_start = self.stats["ep_last_training"]-1
        train_end = self.stats["episode_last_switch_to_evaluation"]-1  # -2+1
        eval_start = self.stats["episode_last_switch_to_evaluation"]-1
        eval_end = len(self.stats["ep_rewards_calc"])  # -1+1=0

        train_rewards_arr = np.asarray(self.stats["ep_rewards_true"])[
            train_start:train_end]
        eval_rewards_arr = np.asarray(self.stats["ep_rewards_true"])[
            eval_start:eval_end]

        training_matches = np.asarray(self.stats["ep_wins"])[
            train_start:train_end]
        eval_matches = np.asarray(self.stats["ep_wins"])[
            eval_start:eval_end]

        avg_train_rewards = (np.sum(train_rewards_arr)
                             / len(train_rewards_arr))
        avg_eval_rewards = (np.sum(eval_rewards_arr)
                            / len(eval_rewards_arr))

        train_wins, train_lost, train_draws = self._calculateWLDRates(
            training_matches, eval_start, eval_end)

        eval_wins, eval_lost, eval_draws = self._calculateWLDRates(
            eval_matches, eval_start, eval_end)

        hist_index = 1 if self.current_opp_weak else 0
        hist_index += 2*(self.current_opp_type.value-1)

        save_text = ""
        if len(self.stats["ep_eval_results"][hist_index]) == 0:
            past_max_values = [-1000000]
        else:
            past_max_values = np.max(np.asarray(self.stats
                                                ["ep_eval_results"]
                                                [hist_index]))

        if past_max_values < avg_eval_rewards:
            save_text = " - Checkpoint saved"
            self.save("best")
            if (self.agent_register is not None
                    and len(self.stats["ep_eval_results"][hist_index]) > 0
                    and avg_eval_rewards >= 4):
                self.agent_register.add_agent(self.config["filename"], "best")
                save_text = save_text + " - added"

            self.stats["ep_eval_results"][hist_index].append(
                avg_eval_rewards)

        print(("{}: ## Learn(R;W,D,L in %): {:.1f}, {:.0f}, {:.0f}, {:.0f} " +
              "Eval(R;W,D,L in %): {:.1f}, {:.0f}, {:.0f}, {:.0f} {}").format(
                  self.stats["ep_counter"]-1,
                  avg_train_rewards, train_wins, train_draws, train_lost,
                  avg_eval_rewards, eval_wins, eval_draws, eval_lost,
                  save_text))

    def act(self, state):
        if self.mode == Mode.Training:
            if self.current_frame_skip_pos == 0:
                self.current_frame_skip_pos = (
                    self.config["frame_skipping_length"]
                    if self.current_frame_skip_activated else 1)
                action = self.ppo.act(state, self.memory)
                self.current_frame_skip_action = action
            else:
                action = self.current_frame_skip_action

            self.current_frame_skip_pos -= 1
        else:
            action = self.ppo.act(state)

        return action

    def change_frame_skipping_mode(self, activate):
        if self.current_ep_timestep_counter != 0:
            raise Exception("Can't switch mode during episode")
        if self.mode == Mode.Evaluation:
            raise Exception("Can't be activated during evaluation")

        self.current_frame_skip_activated = activate

    def feedback(self, reward, info, done, state):
        # Calculate enginered reward
        reward_calc = self.calculate_reward(reward, info, done)

        # Frame skipping: Return same action
        if (self.current_frame_skip_pos > 0 and
                not done and not self.mode == Mode.Evaluation):
            self.current_frame_skip_reward_calc += reward
            self.current_frame_skip_reward_true += reward_calc
            return

        elif ((self.current_frame_skip_pos == 0 or done)
                and not self.mode == Mode.Evaluation):
            reward_calc = self.current_frame_skip_reward_calc + reward_calc
            reward = self.current_frame_skip_reward_true + reward

            self.current_frame_skip_reward_calc = 0
            self.current_frame_skip_reward_true = 0

        if (self.mode == Mode.Training and
                (self.current_frame_skip_pos == 0 or done)):
            self.memory.rewards.append(reward_calc)
            self.memory.is_terminals.append(done)
            self.timesteps_since_update += 1

        # Increase all temporary statistics numbers
        self.current_ep_timestep_counter += 1
        self.stats["global_timestep_counter"] += 1
        self.current_reward_calc += reward_calc
        self.current_reward_true += reward
        self.current_closeness_puck += info["reward_closeness_to_puck"]
        self.current_touch_puck += info["reward_touch_puck"]
        self.current_puck_direction += info["reward_puck_direction"]
        self.memory.lengths[-1] += 1

        if done or (self.current_ep_timestep_counter
                    % self.config["episode_max_steps"] == 0):
            # Finalize last episode
            self.stats["ep_counter"] += 1
            if self.mode == Mode.Training:
                self.stats["learning_ep_counter"] += 1
            self.stats["episode_mode"].append(self.mode.value)
            self.stats["ep_rewards_calc"].append(self.current_reward_calc)
            self.stats["ep_rewards_true"].append(self.current_reward_true)
            self.stats["ep_closeness_puck"].append(self.current_closeness_puck)
            self.stats["ep_touch_puck"].append(self.current_touch_puck)
            self.stats["ep_puck_direction"].append(self.current_puck_direction)
            self.stats["ep_length"].append(self.current_ep_timestep_counter)
            self.stats["ep_wins"].append(info["winner"])
            self.stats["ep_opp_type"].append(self.current_opp_type)
            self.stats["ep_opp_weak"].append(self.current_opp_weak)

            doc_frame_skip = 1
            if (self.mode == Mode.Evaluation and
                    self.current_frame_skip_activated):
                doc_frame_skip = self.stats["frame_skipping_length"]
            self.stats["ep_frame_skip"].append(doc_frame_skip)

            self.memory.winners.append(info["winner"])
            self.memory.final_states.append(state.reshape(-1))

            # Prepare next expisode: Reset temporary statistics
            self.current_frame_skip_pos = 0
            self.current_reward_calc = 0
            self.current_reward_true = 0
            self.current_ep_timestep_counter = 0
            self.current_closeness_to_puck = 0
            self.current_touch_puck = 0
            self.current_puck_direction = 0

            # Prepare next episode: Set next params
            self.memory.lengths.append(0)

            if self.stats["ep_counter"] % self.config["save_episodes"] == 0:
                self.save()

            if (self.mode == Mode.Training and self.timesteps_since_update >=
                    self.config["update_episodes"]):
                self.update()

    def update(self):
        if self.memory.lengths[-1] == 0:
            del self.memory.lengths[-1]
        self.ppo.update(self.memory)
        self.memory.clear_memory()
        self.memory.lengths.append(0)
        self.timesteps_since_update = 0

    def generate_filename(self, info=""):
        return "checkpoints/{}_{}_{}.pth".format(
            self.config["environment_name"], self.config["filename"], info)

    def save(self, info=""):
        filename = self.generate_filename(info)

        torch.save({
            'policy': self.ppo.policy.state_dict(),
            'policy_old': self.ppo.policy_old.state_dict(),
            'optimizer': self.ppo.optimizer.state_dict(),
            'configuration': self.config,
            'statistics': self.stats,
            'input_normalizer': pickle.dumps(self.ppo.input_normalizer),
            'advantage_normalizer':
                pickle.dumps(self.ppo.advantage_normalizer),
            'reward_normalizer': pickle.dumps(self.ppo.reward_normalizer),
            'memory': self.memory,
        }, filename)

    def load(self, environment_name, filename, info=""):
        filename = "checkpoints/{}_{}_{}.pth".format(
            environment_name, filename, info)
        checkpoint = torch.load(filename)

        self.config = checkpoint["configuration"]
        self.stats = checkpoint["statistics"]
        self.stats = checkpoint["statistics"]

        self.configure_PPO()
        self.memory = checkpoint["memory"]

        self.ppo.policy.load_state_dict(checkpoint["policy"])
        self.ppo.policy_old.load_state_dict(checkpoint["policy_old"])
        self.ppo.optimizer.load_state_dict(checkpoint["optimizer"])
        self.ppo.input_normalizer = pickle.loads(
            checkpoint["input_normalizer"])

        self.ppo.advantage_normalizer = pickle.loads(
            checkpoint["advantage_normalizer"])
        self.ppo.reward_normalizer = pickle.loads(
            checkpoint["reward_normalizer"])

    def calculate_reward(self, reward, info, done):
        if self.mode == Mode.Evaluation:
            return reward
        elif done:
            value = (reward * self.config["weighting_true_reward"]
                     + info["winner"] * self.config["weighting_reward_winner"]
                     + (info["reward_closeness_to_puck"]
                        * self.config["weighting_reward_closeness_puck"])
                     + ((info["reward_touch_puck"]
                        * self.config["weighting_reward_touch_puck"]))
                     + (info["reward_puck_direction"]
                        * self.config["weighting_reward_puck_direction"])
                     - self.config["default_timestep_loss"])
            return value
        else:
            return 0

    def set_opponent(self, opponent_type, opponent_weak):
        if self.current_ep_timestep_counter != 0:
            raise Exception("Can't switch mode during episode")

        self.current_opp_type = opponent_type
        self.current_opp_weak = opponent_weak

    def set_filename(self, filename):
        self.stats["filename"] = filename

    def set_frame_skipping(self, frame_skipping_length,
                           frame_skipping_activated):
        if self.current_ep_timestep_counter != 0:
            raise Exception("Can't switch mode during episode")

        self.config["frame_skipping_length"] = frame_skipping_length
        self.current_frame_skip_activated = frame_skipping_activated

    def set_update_episodes(self, set_update_episodes):
        if self.current_ep_timestep_counter != 0:
            raise Exception("Can't switch mode during episode")

        self.config["update_episodes"] = set_update_episodes

    def set_reward_weights(self, weighting_true_reward,
                           weighting_reward_winner,
                           weighting_reward_closeness_puck,
                           weighting_reward_touch_puck,
                           weighting_reward_puck_direction,
                           default_timestep_loss):
        if self.current_ep_timestep_counter != 0:
            raise Exception("Can't switch mode during episode")

        self.config["weighting_true_reward"] = weighting_true_reward
        self.config["weighting_reward_winner"] = weighting_reward_winner
        self.config["weighting_reward_closeness_puck"] = weighting_reward_closeness_puck
        self.config["weighting_reward_touch_puck"] = weighting_reward_touch_puck
        self.config["weighting_reward_puck_direction"] = weighting_reward_puck_direction
        self.config["default_timestep_loss"] = default_timestep_loss
