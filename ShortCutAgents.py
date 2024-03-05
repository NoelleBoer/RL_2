import random
import numpy as np


class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.action_values = np.zeros((n_states, n_actions))
        pass

    def select_action(self, state):
        best_action = np.argmax(self.action_values)
        policy = np.full_like(self.action_values, (self.epsilon / (self.n_actions - 1)))
        policy[best_action] = 1 - self.epsilon
        a = np.random.choice(self.n_actions, p=policy)
        return a

    def update(self, state, action, reward):
        # TO DO: Add own code
        pass


class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        # TO DO: Add own code
        pass

    def select_action(self, state):
        # TO DO: Add own code
        a = random.choice(range(self.n_actions))  # Replace this with correct action selection
        return a

    def update(self, state, action, reward):
        # TO DO: Add own code
        pass


class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        # TO DO: Add own code
        pass

    def select_action(self, state):
        # TO DO: Add own code
        a = random.choice(range(self.n_actions))  # Replace this with correct action selection
        return a

    def update(self, state, action, reward):
        # TO DO: Add own code
        pass
