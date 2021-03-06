import numpy as np
from enum import Enum
import torch
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F
from Normalizer import Normalizer
from Memory import Memory

device = "cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ZERO_STD = 0.0000001
MAX_STD = 15


class ActivationFunctions(Enum):
    """Options that can be chosen as Activation functions within NN"""
    TANH = 1
    ELU = 2
    SWISH = 3


class Initializers(Enum):
    """Options forthat can be chosen for the initialization of the NN"""
    LECUN_NORMAL = 1
    GLOROT_NORMAL = 2
    ORTHOGONAL = 3
    ORTHOGONAL14 = 4


class BatchModes(Enum):
    """For SHUFFLE_RECOMPUTE_ADVANTAGES the advantages and returns
    are shuffled for every batch at the beginning of the update
    process.  For SHUFFLE this only happens for the first batch."""
    SHUFFLE = 1
    SHUFFLE_RECOMPUTE_ADVANTAGES = 2


class ActorCritic(nn.Module):
    """Source: https://github.com/nikhilbarhate99/PPO-PyTorch"""

    def __init__(self, state_dim, discrete, action_dim, policy_depth,
                 policy_width, value_depth, value_width, activation_function,
                 minimum_std, initial_std, initializer,
                 policy_last_layer_scaler, value_last_layer_scaler):
        super(ActorCritic, self).__init__()

        self._discrete = discrete
        self._action_dim = action_dim
        self._minimum_std = ZERO_STD if ZERO_STD > minimum_std else minimum_std

        # Activation functions used for actor and critic network
        if activation_function == ActivationFunctions.TANH:
            Function = nn.Tanh
        elif activation_function == ActivationFunctions.ELU:
            Function = nn.ELU
        else:
            Function = nn.SiLU

        # Initialize policy network
        # 1. Hidden layers are generated dynamically depending on
        #    parameters policy_depth, policy_width. The last hidden
        #    layer has the width of the action dimension.
        # 2. Initialize dynamically
        # 3. If discrete we need Softmax at the end to get a probability
        #    for each action dimension
        policy_layers = []

        policy_layers.append(nn.Linear(state_dim, policy_width))
        policy_layers.append(Function())

        for i in range(policy_depth-1):
            policy_layers.append(nn.Linear(policy_width, policy_width))
            policy_layers.append(Function())

        if discrete:
            policy_layers.append(nn.Linear(policy_width, action_dim))
        else:
            policy_layers.append(nn.Linear(policy_width, 2*action_dim))

        # No scaling for LeCun_normal
        if initializer == Initializers.GLOROT_NORMAL:
            torch.nn.init.xavier_uniform_(policy_layers[-1].weight,
                                          gain=policy_last_layer_scaler)
        elif (initializer == Initializers.ORTHOGONAL or
              initializer == Initializers.ORTHOGONAL14):
            torch.nn.init.orthogonal_(
                policy_layers[-1].weight, gain=policy_last_layer_scaler)

        if discrete:
            policy_layers.append(nn.Softmax(dim=-1))
        self.actor = nn.Sequential(*policy_layers)

        # Initialize policy network
        # 1. Hidden layers are generated dynamically. The last layer is
        #    1, because the value output is just a scalar.
        # 2. Initialize dynamically

        value_layers = []

        value_layers.append(nn.Linear(state_dim, value_width))
        value_layers.append(Function())

        for i in range(value_depth-1):
            value_layers.append(nn.Linear(value_width, value_width))
            value_layers.append(Function())

        value_layers.append(nn.Linear(value_width, 1))

        # No scaling for LeCun_normal
        if initializer == Initializers.GLOROT_NORMAL:
            torch.nn.init.xavier_uniform_(value_layers[-1].weight,
                                          gain=value_last_layer_scaler)
        elif (initializer == Initializers.ORTHOGONAL or
                initializer == Initializers.ORTHOGONAL14):
            torch.nn.init.orthogonal_(value_layers[-1].weight,
                                      gain=value_last_layer_scaler)

        self.critic = nn.Sequential(*value_layers)

        # Initialze all linear layers except last
        for l in (policy_layers[:-2] + value_layers[:-1]):
            if isinstance(l, nn.Linear):
                # LeCun Normal is already default
                if initializer == Initializers.GLOROT_NORMAL:
                    torch.nn.init.xavier_uniform_(l.weight)
                elif initializer == Initializers.ORTHOGONAL:
                    torch.nn.init.orthogonal_(l.weight)
                elif initializer == Initializers.ORTHOGONAL14:
                    torch.nn.init.orthogonal_(l.weight, gain=1.4)

        # Action std parameter indepdenent of observations.
        self._action_std_log = torch.nn.Parameter(
            np.log(initial_std)
            * torch.ones(action_dim, dtype=torch.float32, device=device))
        # Very small std parameter in case of evaluation.
        self._evaluation_std = (
            ZERO_STD
            * torch.ones((action_dim,), dtype=torch.float32, device=device))

    def forward(self, state, evaluation):
        """Compute action distribution"""
        if self._discrete:
            action_probs = self.actor(state)
            return Categorical(action_probs)
        else:
            network_output = self.actor(state)
            action_mean = network_output[:, 0:self._action_dim]

            if evaluation:  # Mode.Evaluation: Fixed std
                action_std = self._evaluation_std.detach()
            else:  # Mode.Training: Use parametric std + clip
                action_std = torch.exp(
                    self._action_std_log).expand_as(action_mean)
                action_std = torch.clamp(
                    action_std, self._minimum_std, MAX_STD)

            cov_mat = torch.diag_embed(action_std**2)

            return MultivariateNormal(action_mean, cov_mat)

    def act(self, state, memory=None):
        # Memory is None => Evaluation
        # Memory is not None => Training
        evaluation = memory is None

        dist = self(state, evaluation)
        action = dist.sample()

        if evaluation:
            torch.no_grad()
        else:
            # Save taken action, states, log probabilities for model
            # update
            action_logprob = dist.log_prob(action)
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)

        return action.cpu().numpy()

    def evaluate(self, state, action):
        dist = self(state, False)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, discrete, action_dim, K_epochs, eps_clip,
                 policy_depth, policy_width, value_depth, value_width,
                 activation_function, minimum_std, initial_std,
                 policy_last_layer_scaler, value_last_layer_scaler,
                 initializer, lbda, mini_batch_size, batch_mode, gamma,
                 handle_abandoned, frame_skipping_length, optimizer_lr,
                 optimizer_weight_decay, optimizer_momentum, optimizer_epsilon,
                 advantage_normalization, reward_normalization,
                 input_normalization, gradient_clipping,
                 input_clipping_max_abs_value):

        self._discrete = discrete

        self._K_epochs = K_epochs
        self._eps_clip = eps_clip

        self._lbda = lbda

        self._mini_batch_size = mini_batch_size
        self._batch_mode = batch_mode

        self._gamma = gamma
        self._handle_abandoned = handle_abandoned
        self._frame_skipping_length = frame_skipping_length

        self._advantage_normalization = advantage_normalization
        self._reward_normalization = reward_normalization
        self._input_normalization = input_normalization
        self.input_normalizer = Normalizer(state_dim,
                                           input_clipping_max_abs_value)
        self.advantage_normalizer = Normalizer(1, 20)
        self.reward_normalizer = Normalizer(1, 20)

        self._gradient_clipping = gradient_clipping
        self._input_clipping_max_abs_value = input_clipping_max_abs_value

        self.policy = ActorCritic(state_dim, discrete, action_dim,
                                  policy_depth, policy_width, value_depth,
                                  value_width, activation_function,
                                  minimum_std, initial_std, initializer,
                                  policy_last_layer_scaler,
                                  value_last_layer_scaler).to(device)

        self.policy_old = ActorCritic(state_dim, discrete, action_dim,
                                      policy_depth, policy_width, value_depth,
                                      value_width, activation_function,
                                      minimum_std, initial_std, initializer,
                                      policy_last_layer_scaler,
                                      value_last_layer_scaler).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=optimizer_lr,
                                          weight_decay=optimizer_weight_decay,
                                          betas=(optimizer_momentum, 0.999),
                                          eps=optimizer_epsilon)

        self._mse_loss = nn.MSELoss()

    def act(self, state, memory=None):
        if self._input_normalization:
            state = self.input_normalizer.add_and_normalize(state)

        if self._discrete:
            state = torch.from_numpy(state).float().to(device)
            return self.policy_old.act(state, memory)
        else:
            state = np.asarray(state)
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            return np.tanh(self.policy_old.act(state, memory).flatten())

    def update(self, memory):
        """
        1. Reward normalization (optional)
        2. Convert to tensor
        3. For each epoch
        3.1 For each batch
        3.1.1 Advantage normalization (optional)
        3.1.2 PPO Update
        3.2. Copy current policy to old policy
        """
        if self._reward_normalization:
            new_rewards = np.zeros(len(memory.rewards))
            for i, r in enumerate(memory.rewards):
                new_rewards[i] = self.reward_normalizer.add_and_normalize(r)
            memory.rewards = new_rewards.tolist()

        # convert to tensor
        if self._discrete:
            old_states = torch.stack(memory.states).to(device).detach()
            old_actions = torch.stack(memory.actions).to(device).detach()
            old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        else:
            old_states = torch.squeeze(torch.stack(memory.states).to(device),
                                       1).detach()
            old_actions = torch.squeeze(torch.stack(memory.actions).to(device),
                                        1).detach()
            old_logprobs = torch.squeeze(torch.stack(memory.logprobs),
                                         1).to(device).detach()

        for i in range(self._K_epochs):
            # Recompute advantages, returns
            if (i == 0 or self._batch_mode ==
                    BatchModes.SHUFFLE_RECOMPUTE_ADVANTAGES):
                advantages, returns = (
                    self.compute_advantages_and_returns(memory))

            for states_, actions_, old_logprobs_, returns_, advantages_ in \
                self.generate_batch_iterations(old_states, old_actions,
                                               old_logprobs, returns,
                                               advantages):
                if self._advantage_normalization:
                    new_advantages = np.zeros(len(advantages_))
                    for i, adv in enumerate(advantages_):
                        new_advantages[i] = (
                            self.advantage_normalizer.add_and_normalize(adv))
                    advantages_ = torch.from_numpy(
                        new_advantages).float().to(device)

                # Evaluating old actions and values under current policy:
                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    states_, actions_)

                # Calculate the change in probability.
                # Ratio more numerically stable.
                ratios = torch.exp(logprobs - old_logprobs_)

                # Finding Surrogate Loss:
                surr_policy_1 = ratios * advantages_
                surr_policy_2 = torch.clamp(
                    ratios, 1-self._eps_clip, 1+self._eps_clip) * advantages_

                # Calculate value loss
                value_loss = self._mse_loss(state_values, returns_)

                # Combine the surrogate policy loss and value loss
                loss = (-torch.min(surr_policy_1, surr_policy_2)
                        + 0.5*value_loss
                        - 0.01*dist_entropy)

                # Take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                if self._gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self._gradient_clipping)
                self.optimizer.step()

        # Copy new weights into old policy after each epoch:
        self.policy_old.load_state_dict(self.policy.state_dict())

    # Source: https://github.com/higgsfield/RL-Adventure-2/
    # blob/master/3.ppo.ipynb
    def generate_batch_iterations(self, states, actions, log_probs, returns,
                                  advantages):
        batch_size = states.size(0)
        for _ in range(batch_size // self._mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, self._mini_batch_size)
            yield (states[rand_ids, :], actions[rand_ids, :],
                   log_probs[rand_ids], returns[rand_ids],
                   advantages[rand_ids])

    def compute_advantages_and_returns(self, memory):
        """Compute Advantages, Returns using GAE
        1. Calculate values (V_t) of visited states
        2. For each episode
        2.1. Delta = 0
        2.2. GAE = 0
        2.3 For each timeste t in the episode (from last to first):
        2.3.1. Delta = Reward_t + Gamma * V_{t+1} - V_t
        2.3.2. GAE = Delta + Gamma * Lambda * GAE
        2.3.3. Returns_t = GAE + V_t
        3. Advantage = Returns - V = GAE
        """
        states = torch.stack(memory.states).to(device)
        values = self.policy.critic(states).flatten().float().detach().cpu()
        values = values.numpy()

        final_states = torch.FloatTensor(memory.final_states).to(device)
        final_values = self.policy.critic(
            final_states).flatten().detach().cpu().numpy()

        returns = np.zeros(len(memory.actions), dtype=float)
        index = len(returns)-1

        step_reverse_counter = memory.lengths[-1]
        reverse_index = len(memory.lengths)-1
        nextValue = final_values[reverse_index]
        mask = np.concatenate((np.ones(step_reverse_counter), np.asarray([0])))
        gae = 0
        last_step = True

        for step in reversed(range(len(memory.rewards))):
            if step_reverse_counter == 0:
                reverse_index -= 1
                nextValue = final_values[reverse_index]
                if reverse_index >= 0:
                    step_reverse_counter = memory.lengths[reverse_index]
                mask = np.concatenate((np.ones(step_reverse_counter),
                                       np.asarray([0])))
                gae = 0
                last_step = True

            reward = memory.rewards[step]

            if last_step and not self._handle_abandoned:
                if len(memory.winners) >= reverse_index+1:
                    if memory.winners[reverse_index] == 0:
                        reward = values[step]
                else:
                    reward = values[step]

            delta = (reward
                     + (self._gamma**self._frame_skipping_length
                        * nextValue*mask[step_reverse_counter])
                     - values[step])
            gae = delta + (self._gamma ** self._frame_skipping_length
                           * self._lbda
                           * mask[step_reverse_counter]
                           * gae)

            returns[index] = gae + values[step]
            index -= 1

            step_reverse_counter -= 1
            last_step = False
            nextValue = values[step]

        advantages = returns - values

        advantages = torch.from_numpy(advantages).float().to(device)
        returns = torch.from_numpy(np.asarray(returns)).float().to(device)
        return advantages, returns
