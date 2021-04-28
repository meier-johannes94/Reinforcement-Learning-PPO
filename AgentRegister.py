import numpy as np
import laserhockey.hockey_env as lh
from Agent import Agent, Mode


class AgentRegister():
    """Allows the agent to also play against his old checkpoints
    (old versions of himself) by sampling the training opponent for each
    training episode from a list of possible opponents.  During
    evaluation only the basic opponent is drawn.  This class manages
    the opponent list.

    Methods:
        add_agent(str, str): Add an agent to the list.
        sample_agent(Mode): Sample an agent for training / evaluation.

    """

    def __init__(self, opponent_weak, opponent_mode):
        """Initializes the register of possible opponents

        Args:
            opponent_weak (Bool):
                True if the basic agent is weak, False if the basic
                agent is normal.  This setting has no influence when an
                old checkpint is sampled later on.
            opponent_mode (OpponentType):
                Definition of the OpponentType for statistical reasons.
        """
        self._opponent_weak = opponent_weak
        self._opponent_mode = opponent_mode

        self._agents = [lh.BasicOpponent(opponent_weak)]
        self._scores = []

    def add_agent(self, filename, info):
        """Adds the agent to the list of possible opponents. Hereby
        the old checkpoint is loaded from the hard disk into memory. The
        evaluation performance is also saved as score. If there are
        more than 20 opponents in the list already, the agent with the
        lowest score is deleted from the list.

        Args:
            filename (str):
                Filename of the checkpoint to be added to the list of
                possible opponents.  The filename is not the complete
                path in the operating system, but Agent_object.filename.
            info (str):
                Defines, whether the checkpoint of the agent was saved,
                because of maximum evaluation performance or because of
                a simple recent saving.
        """
        if len(self._agents) > 20:
            min_index = np.argmin(self.scores)
            del self._scores[min_index]
            del self._agents[min_index+1]

        new_agent = Agent()
        new_agent.load(filename, info)
        new_agent.change_mode(False)

        calculate_history_index = 1 if self._opponent_weak else 0
        calculate_history_index += 2*(self._opponent_mode.value-1)

        self._scores.append(new_agent.stats["ep_eval_results"]
                            [calculate_history_index][-1])
        self._agents.append(new_agent)

    def sample_agent(self, mode):
        """The sampling process has the following logic:
        1. For evaluation: Always take the basic opponent
        2. For training the probability that the basic agent is drawn
        is dependent on the number of agents in the list in total:

         1- 2 list entries => 80% probability for basic agent
         3- 5 list entries => 60% probability for basic agent
         6- 9 list entries => 30% probability for basic agent
        10-14 list entries => 20% probability for basic agent
          >15 list entries => 10% probability for basic agent

        The remaining probability is distributed among the single agents
        using softmax on their evaluation performances.

        Args:
            mode (Mode): Differentiation between Evaluation / Training.

        Returns:
            BasicOpponent / Agent: Opponent for training / evaluation
        """
        if mode == Mode.EVALUATION or len(self._agents) == 1:
            return self._agents[0]
        elif mode == Mode.TRAINING:
            scores = np.asarray(self._scores)
            scores -= np.min(scores) - 1
            logs = np.log(scores)

            prob_basic_agent = 0.8
            if len(self._agents) >= 3:
                prob_basic_agent = 0.6
            elif len(self._agents) >= 6:
                prob_basic_agent = 0.30
            elif len(self._agents) >= 10:
                prob_basic_agent = 0.20
            elif len(self._agents) >= 15:
                prob_basic_agent = 0.10

            p = (1-prob_basic_agent) * self._softmax(logs)
            p = np.insert(p, 0, prob_basic_agent)
            p /= p.sum()
            agent_index = np.random.choice(
                np.arange(0, len(self._agents)), p=p)
            return self._agents[agent_index]

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
