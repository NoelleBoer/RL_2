# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
import numpy as np
import tkinter as tk
from PIL import ImageGrab
from ShortCutEnvironment import ShortcutEnvironment
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent
from Helper import LearningCurvePlot, smooth


def run_repititions_QLearning(n_episodes, n_repetitions=1, epsilon=0.1, alpha=0.1, gamma=1):
    print(f"Running {n_repetitions} repititions of {n_episodes} episodes")
    # Initialise a clean environment
    env = ShortcutEnvironment()
    # Keep track of the average q table over all repititions
    average_q_table = np.zeros((env.state_size(), env.action_size()))
    # Keep track of the average cumulative reward of each episode over all repititions
    average_rewards = np.zeros(n_episodes)
    for rep in range(n_repetitions):
        print(f"Starting repitition {rep+1}")
        # Initialise a clean agent for every repitition
        agent = QLearningAgent(n_actions=env.action_size(), n_states=env.state_size(),
                               epsilon=epsilon, alpha=alpha, gamma=gamma)
        # Keep track of the cumalative reward of each episode
        rewards = np.zeros(n_episodes)
        for ep in range(n_episodes):
            # Get the starting state
            s = env.state()
            while not env.done():
                # Select an action from the policy defined in the agent
                a = agent.select_action(s)
                # Take the action and observe the reward
                r = env.step(a)
                # Add the reward to the total episode rewards
                rewards[ep] += r
                # Get the new state after taking the action
                s_prime = env.state()
                # Update the q table of the agent
                agent.update(s, s_prime, a, r)
                # Set the current state to the new state
                s = s_prime
            # Reset the environment after each episode
            env.reset()
        # Update the average q table over all repititions after finishing all episodes
        average_q_table += 1 / (rep + 1) * (agent.q_table - average_q_table)
        # Update the average cumulative reward of each episode over all repititions after finishing all episodes
        average_rewards += 1 / (rep + 1) * (rewards - average_rewards)
    return average_q_table, average_rewards


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
                pass
        average_q_table += (agent.q_table - average_q_table) / (rep + 1)
    return average_q_table


def print_greedy_actions_tk(Q, file_name):
    root = tk.Tk()

    # Define symbols for each action
    symbols = ['↑', '↓', '←', '→', '⨯']

    labels = []

    for i in range(12):
        for j in range(12):
            max_index = np.argmax(Q[i*12 + j])
            if np.max(Q[i*12 + j]) == 0:
                symbol = symbols[4]
                bg_color = "tomato"
            else:
                symbol = symbols[max_index]
                bg_color = "white"
            if i == 8 and j == 8:
                bg_color = "lime green"
            elif (i == 2 or i == 9) and j == 2:
                bg_color = "cornflower blue"
            label = tk.Label(root, text=symbol, font=("Helvetica", 30),
                             width=2, height=1, borderwidth=1, relief="solid", bg=bg_color)
            label.grid(row=i, column=j)
            labels.append(label)

    def color_path(i, j):
        if 0 <= i < 12 and 0 <= j < 12:
            label = labels[i*12 + j]
            bg_color = label.cget("bg")
            if bg_color == "white":
                label.config(bg="pale green")
            symbol = label.cget("text")
            if symbol == symbols[0]:
                i -= 1
            elif symbol == symbols[1]:
                i += 1
            elif symbol == symbols[2]:
                j -= 1
            elif symbol == symbols[3]:
                j += 1
            else:
                return
            color_path(i, j)

    color_path(2, 2)
    color_path(9, 2)

    root.update_idletasks()
    root.after(100)

    x = root.winfo_rootx()
    y = root.winfo_rooty()
    w = root.winfo_width()
    h = root.winfo_height()
    image = ImageGrab.grab((x, y, x + w, y + h))

    image.save(file_name)

    root.destroy()


def experiment():
    #################
    # 1: Q-learning #
    #################

    # Experiment 1
    n_episodes = 10000
    q_table, rewards = run_repititions_QLearning(n_episodes)
    print_greedy_actions_tk(q_table, file_name="qlearning_path_ep_10000.png")

    # Experiment 2
    smoothing_window = 31
    n_repetitions = 100
    n_episodes = 1000
    q_table, rewards = run_repititions_QLearning(n_episodes, n_repetitions)
    print_greedy_actions_tk(q_table, file_name="qlearning_path_rep_100_ep_10000.png")
    plot = LearningCurvePlot("Q-learning learning curve")
    plot.add_curve(smooth(rewards, window=smoothing_window), label="Q-learning")
    plot.save(name="qlearning_learning_curve.png")

    ############
    # 2: SARSA #
    ############

    #####################
    # 3: Stormy weather #
    #####################

    #####################
    # 4: Expected SARSA #
    #####################


if __name__ == '__main__':
    experiment()
