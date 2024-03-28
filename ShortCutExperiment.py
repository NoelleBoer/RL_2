# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
import numpy as np
import tkinter as tk
from PIL import ImageGrab
from ShortCutEnvironment import ShortcutEnvironment, WindyShortcutEnvironment
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent
from Helper import LearningCurvePlot, smooth


def run_repititions_qlearning(n_episodes, n_repetitions, epsilon=0.1, alpha=0.1, gamma=1, windy=False):
    print("Running repititions with Q-learning using the following settings:")
    print(locals())
    if windy:  # Initialise a clean environment
        env = WindyShortcutEnvironment()
    else:
        env = ShortcutEnvironment()
    average_q_table = np.zeros((env.state_size(), env.action_size()))
    average_rewards = np.zeros(n_episodes)
    for rep in range(n_repetitions):
        print(f"Running repitition {rep+1}", end="\r")
        agent = QLearningAgent(n_actions=env.action_size(), n_states=env.state_size(),
                               epsilon=epsilon, alpha=alpha, gamma=gamma)  # Initialise a clean agent
        rewards = np.zeros(n_episodes)  # Keep track of the cumalative reward of each episode
        for ep in range(n_episodes):
            s = env.state()  # Get the starting state
            while not env.done():
                a = agent.select_action(s)  # Select the action to be performed from the state
                r = env.step(a)  # Take the action and observe the reward
                rewards[ep] += r  # Add the reward to the total episode rewards
                s_prime = env.state()  # Get the new state after taking the action
                agent.update(s, a, r, s_prime)  # Update the q table of the agent
                s = s_prime  # Set the current state to the new state
            env.reset()  # Reset the environment after each episode
        # Update the average q table over all repititions after finishing all episodes
        average_q_table += 1 / (rep + 1) * (agent.q_table - average_q_table)
        # Update the average cumulative reward of each episode over all repititions after finishing all episodes
        average_rewards += 1 / (rep + 1) * (rewards - average_rewards)
    return average_q_table, average_rewards


def run_repititions_sarsa(n_episodes, n_repetitions=1, epsilon=0.1, alpha=0.1, gamma=1, windy=False):
    print("Running repititions with SARSA using the following settings:")
    print(locals())
    if windy:  # Initialise a clean environment
        env = WindyShortcutEnvironment()
    else:
        env = ShortcutEnvironment()
    average_q_table = np.zeros((env.state_size(), env.action_size()))
    average_rewards = np.zeros(n_episodes)
    for rep in range(n_repetitions):
        print(f"Running repitition {rep+1}", end="\r")
        agent = SARSAAgent(n_actions=env.action_size(), n_states=env.state_size(),
                           epsilon=epsilon, alpha=alpha, gamma=gamma)  # Initialise a clean agent
        rewards = np.zeros(n_episodes)  # Keep track of the cumalative reward of each episode
        for ep in range(n_episodes):
            s = env.state()  # Get the starting state
            a = agent.select_action(s)  # Select the initial action for the starting state
            while not env.done():
                r = env.step(a)  # Take the action and observe the reward
                rewards[ep] += r  # Add the reward to the total episode rewards
                s_prime = env.state()  # Get the next state after taking the action
                a_prime = agent.select_action(s)  # Select the action to be performed in the next state
                agent.update(s, a, r, s_prime, a_prime)  # Update the q table of the agent
                s = s_prime  # Set the current state to the next state
                a = a_prime  # Set the current action to the next action
            env.reset()  # Reset the environment after each episode
        # Update the average q table over all repititions after finishing all episodes
        average_q_table += 1 / (rep + 1) * (agent.q_table - average_q_table)
        # Update the average cumulative reward of each episode over all repititions after finishing all episodes
        average_rewards += 1 / (rep + 1) * (rewards - average_rewards)
    return average_q_table, average_rewards


def run_repititions_expectedsarsa(n_episodes, n_repetitions, epsilon=0.1, alpha=0.1, gamma=1):
    print("Running repititions with Expected SARSA using the following settings:")
    print(locals())
    env = ShortcutEnvironment()  # Initialise a clean environment
    average_q_table = np.zeros((env.state_size(), env.action_size()))
    average_rewards = np.zeros(n_episodes)
    for rep in range(n_repetitions):
        print(f"Running repitition {rep+1}", end="\r")
        agent = ExpectedSARSAAgent(n_actions=env.action_size(), n_states=env.state_size(),
                                   epsilon=epsilon, alpha=alpha, gamma=gamma)  # Initialise a clean agent
        rewards = np.zeros(n_episodes)  # Keep track of the cumalative reward of each episode
        for ep in range(n_episodes):
            s = env.state()  # Get the starting state
            while not env.done():
                a = agent.select_action(s)  # Select the action to be performed from the state
                r = env.step(a)  # Take the action and observe the reward
                rewards[ep] += r  # Add the reward to the total episode rewards
                s_prime = env.state()  # Get the new state after taking the action
                agent.update(s, a, r, s_prime)  # Update the q table of the agent
                s = s_prime  # Set the current state to the new state
            env.reset()  # Reset the environment after each episode
        # Update the average q table over all repititions after finishing all episodes
        average_q_table += 1 / (rep + 1) * (agent.q_table - average_q_table)
        # Update the average cumulative reward of each episode over all repititions after finishing all episodes
        average_rewards += 1 / (rep + 1) * (rewards - average_rewards)
    return average_q_table, average_rewards


def print_greedy_actions_tk(Q, file_name):
    root = tk.Tk()
    symbols = ['↑', '↓', '←', '→', '⨯']  # Define symbols for each action
    labels = []
    for i in range(12):
        for j in range(12):
            max_index = np.argmax(Q[i*12 + j])
            if np.max(Q[i*12 + j]) == 0:
                if not np.any(Q[i*12 + j]):
                    symbol = symbols[4]
                else:
                    symbol = symbols[max_index]
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


def experiment(q_leaning, sarsa, stormy_weather, expected_sarsa):
    #################
    # 1: Q-learning #
    #################

    if q_leaning:
        # Experiment 1
        n_repetitions = 1
        n_episodes = 10000
        q_table, rewards = run_repititions_qlearning(n_episodes=n_episodes, n_repetitions=n_repetitions)
        print_greedy_actions_tk(q_table, file_name="qlearning_path.png")

        # Experiment 2
        smoothing_window = 31
        n_repetitions = 100
        n_episodes = 1000
        plot = LearningCurvePlot("Q-learning learning curve")
        for alpha in [0.01, 0.1, 0.5, 0.9]:
            q_table, rewards = run_repititions_qlearning(
                n_episodes=n_episodes, n_repetitions=n_repetitions, alpha=alpha)
            plot.add_curve(smooth(rewards, window=smoothing_window), label=f"α = {alpha}")
        plot.save(name="qlearning_learning_curve.png")

    ############
    # 2: SARSA #
    ############

    if sarsa:
        # Experiment 1
        n_repetitions = 1
        n_episodes = 10000
        q_table, rewards = run_repititions_sarsa(n_episodes=n_episodes, n_repetitions=n_repetitions)
        print_greedy_actions_tk(q_table, file_name="sarsa_path.png")

        # Experiment 2
        smoothing_window = 31
        n_repetitions = 100
        n_episodes = 1000
        plot = LearningCurvePlot("SARSA learning curve")
        for alpha in [0.01, 0.1, 0.5, 0.9]:
            q_table, rewards = run_repititions_sarsa(n_episodes=n_episodes, n_repetitions=n_repetitions, alpha=alpha)
            plot.add_curve(smooth(rewards, window=smoothing_window), label=f"α = {alpha}")
        plot.save(name="sarsa_learning_curve.png")

    #####################
    # 3: Stormy weather #
    #####################

    if stormy_weather:
        # Experiment 1
        n_repetitions = 1
        n_episodes = 10000
        q_table, rewards = run_repititions_qlearning(n_episodes=n_episodes, n_repetitions=n_repetitions, windy=True)
        print_greedy_actions_tk(q_table, file_name="qlearning_path_windy.png")

        # Experiment 2
        n_repetitions = 1
        n_episodes = 10000
        q_table, rewards = run_repititions_sarsa(n_episodes=n_episodes, n_repetitions=n_repetitions, windy=True)
        print_greedy_actions_tk(q_table, file_name="sarsa_path_windy.png")

    #####################
    # 4: Expected SARSA #
    #####################

    if expected_sarsa:
        # Experiment 1
        n_repetitions = 1
        n_episodes = 10000
        q_table, rewards = run_repititions_expectedsarsa(n_episodes=n_episodes, n_repetitions=n_repetitions)
        print_greedy_actions_tk(q_table, file_name="expectedsarsa_path.png")

        # Experiment 2
        smoothing_window = 31
        n_repetitions = 100
        n_episodes = 1000
        plot = LearningCurvePlot("Expected SARSA learning curve")
        for alpha in [0.01, 0.1, 0.5, 0.9]:
            q_table, rewards = run_repititions_expectedsarsa(
                n_episodes=n_episodes, n_repetitions=n_repetitions, alpha=alpha)
            plot.add_curve(smooth(rewards, window=smoothing_window), label=f"α = {alpha}")
        plot.save(name="expectedsarsa_learning_curve.png")


if __name__ == '__main__':
    experiment(q_leaning=False, sarsa=False, stormy_weather=False, expected_sarsa=True)
