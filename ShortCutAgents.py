import numpy as np
import random


class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon, alpha=0.1, gamma=1):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((n_states, n_actions))  # Initialize Q-table with zeros
        pass

    def select_action(self, state):
        # Find the index of the current best arm with the highest mean
        best_action = np.argmax(self.q_table[state])
        # Initialise the policy with the probability of not selecting the current best arm
        policy = np.full_like(self.q_table[state], (self.epsilon / (self.n_actions - 1)))
        # Set the probabily of selecting the current best arm
        policy[best_action] = 1 - self.epsilon
        # Sample from the arms using the policy
        a = np.random.choice(self.n_actions, p=policy)
        return a

    def update(self, state, next_state, action, reward):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action] - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_target
        pass


class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=0.99):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((n_states, n_actions))  # Initialize Q-table with zeros
        pass

    def select_action(self, state):
        # Find the index of the current best arm with the highest mean
        best_action = np.argmax(self.q_table[state])
        # Initialise the policy with the probability of not selecting the current best arm
        policy = np.full_like(self.q_table[state], (self.epsilon / (self.n_actions - 1)))
        # Set the probabily of selecting the current best arm
        policy[best_action] = 1 - self.epsilon
        # Sample from the arms using the policy
        a = np.random.choice(self.n_actions, p=policy)
        return a

    def update(self, state, action, reward, next_state, next_action):
        td_target = reward + self.gamma * self.q_table[next_state, next_action]
        td_delta = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_delta
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
        # Find the index of the current best arm with the highest mean
        best_action = np.argmax(self.q_table[state])
        # Initialise the policy with the probability of not selecting the current best arm
        policy = np.full_like(self.q_table[state], (self.epsilon / (self.n_actions - 1)))
        # Set the probabily of selecting the current best arm
        policy[best_action] = 1 - self.epsilon
        # Sample from the arms using the policy
        a = np.random.choice(self.n_actions, p=policy)
        return a

    def update(self, state, action, reward, next_state):
        next_action = self.select_action(next_state)  # Used for Expected SARSA update, but action not actually taken
        next_action_probs = np.ones(self.n_actions) * self.epsilon / self.n_actions
        next_action_probs[np.argmax(self.q_table[next_state])] += (1.0 - self.epsilon)
        expected_q = np.dot(self.q_table[next_state], next_action_probs)
        self.q_table[state, action] += self.alpha * (reward + self.gamma * expected_q - self.q_table[state, action])
        pass
