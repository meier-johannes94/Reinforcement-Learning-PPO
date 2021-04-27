import numpy as np
import laserhockey.hockey_env as lh
from Agent import Agent, Mode


class AgentRegister():
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
            new_agent.statistics["episode_eval_results"]
            [calculate_history_index][-1])
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
