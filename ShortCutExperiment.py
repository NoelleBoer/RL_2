# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
import numpy as np
from ShortCutEnvironment import ShortcutEnvironment
from ShortCutAgents import QLearningAgent


def run_repititions(n_episodes, n_repetitions, epsilon=0.1, alpha=0.1, gamma=1):
    for rep in range(n_repetitions):
        env = ShortcutEnvironment()
        agent = QLearningAgent(n_actions=env.action_size(), n_states=env.state_size(),
                               epsilon=epsilon, alpha=alpha, gamma=gamma)

        for ep in range(n_episodes):
            s = env.state()
            while not env.done():
                a = agent.select_action(s)
                r = env.step(a)
                s_prime = env.state()
                agent.update(s, s_prime, a, r)
                s = s_prime


def print_greedy_actions(Q):
    greedy_actions = np.argmax(Q, 1).reshape((12, 12))
    print_string = np.zeros((12, 12), dtype=str)
    print_string[greedy_actions == 0] = '^'
    print_string[greedy_actions == 1] = 'v'
    print_string[greedy_actions == 2] = '<'
    print_string[greedy_actions == 3] = '>'
    print_string[np.max(Q, 1).reshape((12, 12)) == 0] = '0'
    line_breaks = np.zeros((12, 1), dtype=str)
    line_breaks[:] = '\n'
    print_string = np.hstack((print_string, line_breaks))
    print(print_string.tobytes().decode('utf-8'))


if __name__ == '__main__':
    run_repititions(n_episodes=1, n_repetitions=1)
