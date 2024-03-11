import numpy as np
import random

class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon, alpha=0.1, gamma=0.99):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((n_states, n_actions))  # Initialize Q-table with zeros
        pass

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(range(self.n_actions))  # Exploration
        else:
            action = np.argmax(self.q_table[state])  # Exploitation
        return action

    def update(self, state, action, reward):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        pass

class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon, alpha=0.1, gamma=0.99):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((n_states, n_actions))  # Initialize Q-table with zeros
        pass

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(range(self.n_actions))  # Exploration: random action
        else:
            action = np.argmax(self.q_table[state])  # Exploitation: best action from Q-table
        return action

    def update(self, state, action, reward):
        td_target = reward + self.gamma * self.q_table[next_state, next_action]
        td_delta = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_delta  # Update Q-value
        pass

class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon, alpha=0.1, gamma=0.99):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((n_states, n_actions))  # Initialize Q-table with zeros
        pass

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(range(self.n_actions))  # Exploration
        else:
            action = np.argmax(self.q_table[state])  # Exploitation
        return action

    def update(self, state, action, reward):
        policy = np.ones(self.n_actions) * self.epsilon / self.n_actions
        best_next_action = np.argmax(self.q_table[next_state])
        policy[best_next_action] += (1.0 - self.epsilon)
        expected_q = np.dot(self.q_table[next_state], policy)
        self.q_table[state, action] += self.alpha * (reward + self.gamma * expected_q - self.q_table[state, action])
        pass
