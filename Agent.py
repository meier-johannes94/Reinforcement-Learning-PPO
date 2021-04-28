import numpy as np
import torch
import pickle
from enum import Enum
from PPO import PPO, Memory, ActivationFunctions, Initializers, BatchModes


class Mode(Enum):
    """Differentiation between training and evaluation.

    Training:
    - Model is updated
    - Exploration (Randomness involved)

    Evaluation:
    - Model is not updated
    - Used to assess, whether to save a new checkpoint, because of
    record high evaluation performance
    - No exploration
    """
    TRAINING = 1
    EVALUATION = 2


class OpponentType(Enum):
    """Three modes supported by the hockey environment.

    NORMAL allows for longer training.
    SHOOTING, NORMAL allows training / evaluation only for their
    specific task."""

    DEFENDING = 1
    SHOOTING = 2
    NORMAL = 3


class Agent:
    """
    The agent combines the training, evaluation, statistics, saving,
    loading and adding to checkpoint list in one class.

    Attributes:
        mode : Mode
            Current mode of the agent (Training or Evaluation)
        config : dict
            Settings associated with this agent
        op_type : OpponentType
            Current opponent type of evaluation / training
        op_weak : bool
            Current basic opponent is weak / normal
        frame_skipping_activated : bool
            Frame-skipping is activated in this episode
        frame_skipping_length : unsigned int
            Frame skipping length

    Methods:
        configure(...): Defines the configuration of the agent
        act(arr): Return the action for the state
        feedback(double, dict, bool):
            Process feedback from the environment for the taken action
        change_mode(bool): Switch between training and evaluation mode
        save(str): Save the model under its filename in /checkpoints
        load(str): Load the model by its filename from /checkpoints

    """

    def __init__(self, agent_register=None):
        """Constructor of the agent

        Args:
            agent_register ([AgentRegister], optional):
                If a valid agent register is handed over, then a check-
                point of the current will be added to the registry if
                a checkpoint reaches record high evaluation performance
                and if the mean evaluation performance is also above 4.
                Defaults to None.
        """
        self._mode = Mode.TRAINING
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

    def configure(self, episode_max_steps, filename, save_episodes, K_epochs,
                  eps_clip, policy_depth, policy_width, value_depth,
                  value_width, activation_function, minimum_std, initial_std,
                  policy_last_layer_scaler, value_last_layer_scaler,
                  initializer, lbda, mini_batch_size, batch_mode,
                  update_timesteps, gamma, handle_abandoned,
                  frame_skipping_length, optimizer_lr, optimizer_weight_decay,
                  optimizer_momentum, optimizer_epsilon,
                  advantage_normalization, reward_normalization,
                  input_normalization, gradient_clipping,
                  input_clipping_max_abs_value, weighting_true_reward,
                  weighting_reward_winner, weighting_reward_closeness_puck,
                  weighting_reward_touch_puck, weighting_reward_puck_direction,
                  default_timestep_loss):
        """Configure a new agent.

        Args:
            episode_max_steps (unsigned int):
                Maximum number of timesteps the agant trains within an
                epsiode before it is interrupted.  230 is the default
                value of the environment.
            filename (str):
                Filename under which the checkpoint is saved in the
                /checkpints folder.  Defaults to "".
            save_episodes (unsigned int):
                After every [save_episodes] the current state of the
                agent is saved.
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
            minimum_std (float):
                Lower limit for the standard deviation used to sample
                actions.  Lower standard deviations are cut off
            initial_std (float):
                Initial standard deviation used to sample actions
            policy_last_layer_scaler (float):
                Scaler by which the last weights of the policy network
                are multiplied with after initialization
            value_last_layer_scaler (float):
                Scaler by which the last weights of the policy network
                are multiplied with after initialization
            initializer (Initializers):
                Initializers for the linear layers in the policy and
                actor network
            lbda (float):
                Lambda parameter of the PPO algorithm
            mini_batch_size (unsigned int):
                Size of the mini-batch for updates of the actor/policy
                and critic/value network
            batch_mode (BatchModes):
                For SHUFFLE_RECOMPUTE_ADVANTAGES the advantages and
                returns are shuffled for every batch at the beginning of
                the update process.  For SHUFFLE this only happens for
                the first batch.
            update_timesteps (unsigned int):
                After every [update_timesteps] timesteps of training the
                model is updated.   In case that the current episode is not
                finished, the update starts after the completition of the
                current episode.  The model is also updated before switching
                to evaluation mode.
            gamma (float):
                Discount factor gamma.  Value between 0 and 1.
            handle_abandoned (bool):
                For False the reward assigned to the last timestep of an
                episode is overwritten by the value of that last
                timestep / last state.
            frame_skipping_length (int):
                Frame skipping may lead to fast computation by using a
                calculated action for [frame_skipping_length] timesteps.
                However it might also lead to a worse performance.
            optimizer_lr(float):
                Learning rate of the optimizer
            optimizer_weight_decay (float):
                Weight decay (Regularization) of the optimizer.
            optimizer_momentum (float):
                Momentum of the optimizer
            optimizer_epsilon (float):
                Epsilon of the optimizer
            advantage_normalization (bool):
                If true the advantages of every mini-batch are
                normalized using an online mean and variance calculation
            reward_normalization (bolol):
                If True the collected rewards during trainig time are
                normalized before updating the network. For the
                calculation of mean and std an online algorithm is use
            input_normalization (bool):
                If true, the observations are normalized using an online
                mean and variance calculation
            gradient_clipping (int):
                If not None the gradient is clipped with the here
                specified value.
            input_clipping_max_abs_value (float):
                Input normalization may lead to high values (e.g. at the
                beginning). Hereby normalized values lower than
                -input_clipping_max_abs_value or higher than
                +input_clipping_max_abs_value are cut off.
            weighting_true_reward (float):
                Synthetic reward +=
                [weighting_true_reward] * [rewal_reward]
            weighting_reward_winner (float):
                Synthetic reward +=
                [weighting_reward_winner] * [info[winner]]
            weighting_reward_closeness_puck (float):
                Synthetic reward +=
                [weighting_reward_closeness_puck] *
                [info[closeness_to_puck]]
            weighting_reward_touch_puck (float):
                Synthetic reward +=
                [weighting_reward_touch_puck] *
                [info[touch_puck]]
            weighting_reward_puck_direction (float):
                Synthetic reward +=
                [weighting_reward_puck_direction] *
                [info[puck_direction]]
            default_timestep_loss (float):
                Defines a reward, whose inverted (negative) value is added
                at each timeste p
        """

        self._config = {
            # Environment & general setting
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

            # Training setup
            'mini_batch_size': mini_batch_size,
            'batch_mode': batch_mode,
            'update_timesteps': update_timesteps,

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
            18, False, 4, self._config["K_epochs"], self._config["eps_clip"],
            self._config["policy_depth"], self._config["policy_width"],
            self._config["value_depth"], self._config["value_width"],
            self._config["activation_function"], self._config["minimum_std"],
            self._config["initial_std"],
            self._config["policy_last_layer_scaler"],
            self._config["value_last_layer_scaler"],
            self._config["initializer"], self._config["lbda"],
            self._config["mini_batch_size"],
            self._config["batch_mode"], self._config["gamma"],
            self._config["handle_abandoned"],
            self._config["frame_skipping_length"],
            self._config["optimizer_lr"],
            self._config["optimizer_weight_decay"],
            self._config["optimizer_momentum"],
            self._config["optimizer_epsilon"],
            self._config["advantage_normalization"],
            self._config["reward_normalization"],
            self._config["input_normalization"],
            self._config["gradient_clipping"],
            self._config["input_clipping_max_abs_value"]
        )

    def act(self, state):
        """Finds and returns the corresponding action to the
        observation. When frame skipping is activated the same action
        is repeated [frame_skipping_length] times.

        Args:
            state (Arr): Current state / obsevation of /from the
            environment

        Returns:
            arr: Action the model suggests to take.  During training
            more randomness is involved to encourage exploration.
        """
        if self._mode == Mode.TRAINING:
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
        """Update the statistics / model of the agent. Call this method
        after each timestep to give the agent feedback about his most
        recently taken action.

        Args:
            reward (float): Reward
            info (arr): Info array as provided by the hockey environment
            done (bool):
                True if this was last action corresponds to the last
                action in the current episode. Otherwise false.
            state (arr):
                Observation after the action. This is important if
                the environment is done for this episode.
        """
        # Calculate enginered reward
        reward_calc = self._calculate_reward(reward, info, done)

        # Frame skipping: Return same action
        if (self._current_frame_skip_pos > 0 and
                not done and not self._mode == Mode.EVALUATION):
            self._current_frame_skip_reward_calc += reward
            self._current_frame_skip_reward_true += reward_calc
            return

        elif ((self._current_frame_skip_pos == 0 or done) and
                not self._mode == Mode.EVALUATION):
            reward_calc = self._current_frame_skip_reward_calc + reward_calc
            reward = self._current_frame_skip_reward_true + reward

            self._current_frame_skip_reward_calc = 0
            self._current_frame_skip_reward_true = 0

        if (self._mode == Mode.TRAINING and
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
            if self._mode == Mode.TRAINING:
                self._stats["learning_ep_counter"] += 1
            self._stats["episode_mode"].append(self._mode.value)
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
            if (self._mode == Mode.EVALUATION and
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

            if (self._mode == Mode.TRAINING and self._timesteps_since_update >=
                    self._config["update_timesteps"]):
                self._update()

    def _calculate_reward(self, reward, info, done):
        if self._mode == Mode.EVALUATION:
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
        """If currently in evaluation, switch to training.
        If currently in training, switch to evaluation.

        Args:
            print_results (bool, optional):
                If true, the statistics of the most recent training
                and evaluation cycle are printed in the console. Only
                relevant when switching from evaluation to training.
                Defaults to False.

        Raises:
            Exception: Switching is only allowed after the completition
            of an episode.
        """
        if self._current_ep_timestep_counter != 0:
            raise Exception("Can't switch mode during episode")

        if self._mode == Mode.TRAINING:
            if len(self._memory.actions) > 0:
                self._update()

            self.frame_skipping_activated = False
            self._mode = Mode.EVALUATION
            self._stats["episode_last_switch_to_evaluation"] = \
                self._stats["ep_counter"]

        elif self._mode == Mode.EVALUATION:
            self._compute_statistics()

            self._mode = Mode.TRAINING
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
            if (self._agent_register is not None and
                    len(self._stats["ep_eval_results"][hist_index]) > 0 and
                    avg_eval_rewards >= 4):
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

    def save(self, info=""):
        """Saves the agent under the following path:
        /checkpoints/Hockey-v0_[filename]_[info].pth.
        From there training can be resumed.

        Args:
            info (str, optional):
                String to further differentiate the model. Defaults to "".
        """
        filename = "checkpoints/Hockey-v0_{}_{}.pth".format(
            self._config["filename"], info)

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

    def load(self, filename, info=""):
        """Load the model from
        /checkpoints/Hockey-v0_[filename]_[info].pth.

        Args:
            filename (str): Filename
            info (str, optional):
                Info for differentation. Defaults to "".
        """
        filename = "checkpoints/Hockey-v0_{}_{}.pth".format(
            filename, info)
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
        """Readonly. Configuration of the model.

        Returns:
            Dict:
        """
        return self._config

    @property
    def mode(self):
        """Readonly. Current mode (Evaluation / Training of the model).

        Returns:
            Mode:
        """
        return self._mode

    @property
    def op_type(self):
        """Opponent Type

        Returns:
            OpponentType:
        """
        return self._current_opp_type

    @op_type.setter
    def op_type(self, value):
        """Opponent Type

        Args:
            value (OpponentType):

        Raises:
            Exception: Switching during running episode
        """
        if self._current_ep_timestep_counter != 0:
            raise Exception("Can't switch opponent_type during episode.")
        self._current_opp_type = value

    @property
    def op_weak(self):
        """Weak or normal basic opponent

        Returns:
            bool: True = Weak, False = Normal
        """
        return self._current_opp_weak

    @op_weak.setter
    def op_weak(self, value):
        """Weak or normal basic opponent

        Args:
            value (bool): True = Weak, False = Normal

        Raises:
            Exception: Switching during running episode
        """
        if self._current_ep_timestep_counter != 0:
            raise Exception(("Can't switch opponent weakness "
                             + "during a running episode."))
        self._current_opp_weak = value

    @property
    def filename(self):
        """Filename under which to save checkpoints

        Returns:
            str:
        """
        return self._config["filename"]

    @filename.setter
    def filename(self, value):
        """Filename under which to save checkpoints

        Args:
            value (str):
        """
        self._config["filename"]

    @property
    def frame_skipping_activated(self):
        """Activation status of frame skipping

        Returns:
            bool:
                If true frame skipping is activated. If false
                deactivated.
        """
        return self._current_frame_skip_activated

    @frame_skipping_activated.setter
    def frame_skipping_activated(self, value):
        """Activation status of frame skipping

        Args:
            value (bool):
                If true frame skipping is activated. If false
                deactivated.

        Raises:
            Exception: Switching during evaluation
            Exception: Switching during runing episode
        """
        if self._mode == Mode.EVALUATION:
            raise Exception("Can't be activated during evaluation")
        if self._current_ep_timestep_counter != 0:
            raise Exception(("Can't switch frame skipping skipping activation "
                             + "during a running episode."))
        self._current_frame_skip_activated = value

    @property
    def frame_skipping_length(self):
        """Number of timesteps the same action is taken.

        Returns:
            unsigned int: Frame skipping length
        """
        return self._current_frame_skip_activated

    @frame_skipping_length.setter
    def frame_skipping_length(self, value):
        """Number of timesteps the same action is taken.

        Args:
            value (unsigned int): Frame skipping length

        Raises:
            Exception: Switching during evaluation
            Exception: Switching during runing episode
        """
        if self._mode == Mode.EVALUATION:
            raise Exception("Can't be activated during evaluation")
        if self._current_ep_timestep_counter != 0:
            raise Exception(("Can't switch frame skipping length during "
                             + "a running episode."))
        self._config["frame_skipping_length"] = value
