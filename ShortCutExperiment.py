# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
import numpy as np
import tkinter as tk
from ShortCutEnvironment import ShortcutEnvironment
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent

def run_repititions_QLearning(n_episodes, n_repetitions, epsilon=0.1, alpha=0.1, gamma=1):
    # Initialise a clean environment
    env = ShortcutEnvironment()
    # Keep track of the average q table over all repititions
    average_q_table = np.zeros((env.state_size(), env.action_size()))
    for rep in range(n_repetitions):
        print(f"Starting repitition {rep+1}")
        # Initialise a clean agent for every repitition
        agent = QLearningAgent(n_actions=env.action_size(), n_states=env.state_size(),
                               epsilon=epsilon, alpha=alpha, gamma=gamma)
        for ep in range(n_episodes):
            # Get the starting state
            s = env.state()
            while not env.done():
                # Select an action from the policy defined in the agent
                a = agent.select_action(s)
                # Take the action and observe the reward
                r = env.step(a)
                # Get the new state after taking the action
                s_prime = env.state()
                # Update the q table of the agent
                agent.update(s, s_prime, a, r)
                # Set the current state to the new state
                s = s_prime
            # Reset the environment after each episode and each repitition
            env.reset()
        # Update the average q table over all repititions after finishing al episodes
        average_q_table += 1 / (rep + 1) * (agent.q_table - average_q_table)
    return average_q_table

def run_repititions_SARSA(n_episodes, n_repetitions, epsilon=0.1, alpha=0.1, gamma=1):
    # Initialise a clean environment
    env = ShortcutEnvironment()
    # Keep track of the average q table over all repititions
    average_q_table = np.zeros((env.state_size(), env.action_size()))
    for rep in range(n_repetitions):
        print(f"Starting repitition {rep+1}")
        # Initialise a clean agent for every repitition
        agent = SARSAAgent(n_actions=env.action_size(), n_states=env.state_size(),
                               epsilon=epsilon, alpha=alpha, gamma=gamma)
        for ep in range(n_episodes):
            # Get the starting state
            s = env.state()
            # Choose the initial action based on the current state
            a = agent.select_action(s)
            while not env.done():
                # Take the action and observe the reward
                r = env.step(a)  # Assuming env.step returns only the reward
                # Get the new state after taking the action
                s_prime = env.state()
                # Select the next action from the next state using the policy defined in the agent
                a_prime = agent.select_action(s_prime)
                # Update the Q table of the agent
                agent.update(s, a, r, s_prime, a_prime)
                # Update the state and action for the next iteration
                s, a = s_prime, a_prime
            # Reset the environment after each episode
            env.reset()
        # Update the average q table over all repititions after finishing al episodes
        average_q_table += 1 / (rep + 1) * (agent.q_table - average_q_table)
    return average_q_table

def run_repititions_ESARSA(n_episodes, n_repetitions, epsilon=0.1, alpha=0.1, gamma=1):
    env = ShortcutEnvironment()
    average_q_table = np.zeros((env.state_size(), env.action_size()))
    for rep in range(n_repetitions):
        agent = ExpectedSARSAAgent(env.state_size(), env.action_size(), alpha, gamma, epsilon)
        for episode in range(n_episodes):
            state = env.reset()
            while not env.done():
                # iets
        average_q_table += (agent.q_table - average_q_table) / (rep + 1)
    return average_q_table

def print_greedy_actions_tk(Q):
    root = tk.Tk()
    root.title("Greedy Actions")

    # Define symbols for each action
    symbols = ['↑', '↓', '←', '→']

    for i in range(12):
        for j in range(12):
            max_index = np.argmax(Q[i*12 + j])
            if np.max(Q[i*12 + j]) == 0:
                symbol = '⨯'
                label_color = 'red'
            else:
                symbol = symbols[max_index]
                label_color = 'black'
            if i == 8 and j == 8:
                label_color = 'green'
            elif (i == 2 or i == 9) and j == 2:
                label_color = 'blue'
            label = tk.Label(root, text=symbol, font=("Helvetica", 16),
                             width=2, height=1, borderwidth=1, relief="solid", fg=label_color)
            label.grid(row=i, column=j)

    root.mainloop()


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
    q = run_repititions_ESARSA(n_episodes=1000, n_repetitions=5)
    print_greedy_actions_tk(q)
