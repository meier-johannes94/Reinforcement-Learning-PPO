import numpy as np
from importlib import reload
import laserhockey.hockey_env as h_env
import gym
from Agent import Agent, Mode


def test_agent(number_games, checkpoint_filename, render,
               checkpoint_info="best", weak_opponent=False,
               checkpoint_selfplay_filename=None,
               checkpoint_selfplay_info="best",
               print_results=True):

    env = h_env.HockeyEnv()
    if checkpoint_selfplay_filename is None:
        opponent = h_env.BasicOpponent(weak_opponent)
    else:
        opponent = Agent()
        opponent.load(checkpoint_selfplay_filename,
                      checkpoint_selfplay_info)

    agent = Agent()
    agent.load(checkpoint_filename, checkpoint_info)

    game_result = np.zeros(number_games)
    rewards = 0

    for i in range(number_games):
        if agent.mode == Mode.TRAINING:
            agent.change_mode(False)

        obs = env.reset()
        obs_agent2 = env.obs_agent_two()
        if render:
            env.render()

        for t in range(600):
            a1 = agent.act(obs)
            a2 = opponent.act(obs_agent2)
            obs, r, d, info = env.step(np.hstack([a1, a2]))
            obs_agent2 = env.obs_agent_two()
            rewards += r
            if render:
                env.render()
            if d:
                game_result[i] = info["winner"]
                break

    env.close()

    if print_results:
        print("Wins: ", np.sum(game_result == 1), "wins")
        print("Draws: ", np.sum(game_result == 0), "draw")
        print("Losses: ", np.sum(game_result == -1), "loss")
        print("Avg. reward: ", np.round(rewards / number_games))
