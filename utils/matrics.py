import numpy as np


class TrainingMetrics:
    """Collects metrics during training"""

    def __init__(self):
        self.episode_rewards = []
        self.max_q_values = []
        self.visited_states = []

    def add_episode(self, episode_reward, max_q_value, number_of_visited_states):
        self.episode_rewards.append(episode_reward)
        self.max_q_values.append(max_q_value)
        self.visited_states.append(number_of_visited_states)

    def get_mean_max_q(self):
        if not self.max_q_values:
            return 0.0

        return float(np.mean(self.max_q_values))
