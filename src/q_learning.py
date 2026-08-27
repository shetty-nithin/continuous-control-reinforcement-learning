import numpy as np
from collections import defaultdict

class QLearningAgent:
    """Tabular Q-learning agent using a sparse dictionary for storing Q-values."""
    def __init__(self, number_of_actions, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, seed=42):
        self.number_of_actions = number_of_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.random_generator = np.random.default_rng(seed)

        self.q_table = defaultdict(
                lambda: np.zeros(self.number_of_actions)
        )

    def select_action(self, state):
        """Select an action using epsilon-greedy exploration."""

        if self.random_generator.random() < self.epsilon:
            return self.random_generator.integers(self.number_of_actions)

        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        current_q_value = self.q_table[state][action]

        if done:
            target = reward
        else:
            best_next_q_value = np.max(self.q_table[next_state])
            target = reward + self.discount_factor * best_next_q_value

        updated_q_value = current_q_value + self.learning_rate * (target - current_q_value) 
        # Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

        self.q_table[state][action] = updated_q_value

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)

