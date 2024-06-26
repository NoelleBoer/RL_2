# Own code added by Daniël Zee (s2063131) and Noëlle Boer (s2505169)

import numpy as np


class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon, alpha, gamma):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((n_states, n_actions))  # Initialize Q-table with zeros
        pass

    def select_action(self, state):  # ϵ-greedy policy
        best_action = np.argmax(self.q_table[state])
        if np.random.random() > self.epsilon:
            action = best_action
        else:
            action = np.random.choice([x for x in range(self.n_actions) if x != best_action])
        return action

    def update(self, state, action, reward, next_state):  # Q-Learning update equation
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_delta = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_delta
        pass


class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon, alpha, gamma):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((n_states, n_actions))  # Initialize Q-table with zeros
        pass

    def select_action(self, state):  # ϵ-greedy policy
        best_action = np.argmax(self.q_table[state])
        if np.random.random() > self.epsilon:
            action = best_action
        else:
            action = np.random.choice([x for x in range(self.n_actions) if x != best_action])
        return action

    def update(self, state, action, reward, next_state, next_action):  # SARSA update equation
        td_target = reward + self.gamma * self.q_table[next_state, next_action]
        td_delta = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_delta
        pass


class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon, alpha, gamma):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((n_states, n_actions))  # Initialize Q-table with zeros
        pass

    def select_action(self, state):  # ϵ-greedy policy
        best_action = np.argmax(self.q_table[state])
        if np.random.random() > self.epsilon:
            action = best_action
        else:
            action = np.random.choice([x for x in range(self.n_actions) if x != best_action])
        return action

    def update(self, state, action, reward, next_state):  # expected SARSA update equation
        best_next_action = np.argmax(self.q_table[next_state])
        next_action_probs = np.full(self.n_actions, self.epsilon / (self.n_actions - 1))
        next_action_probs[best_next_action] = 1.0 - self.epsilon
        expected_q_next_state = np.inner(self.q_table[next_state], next_action_probs)

        td_target = reward + self.gamma * expected_q_next_state
        td_delta = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_delta
        pass
