import gymnasium as gym
import pickle
import numpy as np
import time

from collections import defaultdict

from src.state_discretizer import StateDiscretizer
from src.action_discretizer import ActionDiscretizer
from src.q_learning import QLearningAgent

env = gym.make("HalfCheetah-v5", render_mode="human")
state_discretizer = StateDiscretizer(number_of_bins=7)
action_discretizer = ActionDiscretizer()
agent = QLearningAgent(number_of_actions=action_discretizer.get_number_of_actions(), epsilon=0.0)

with open("q_table.pkl", "rb") as file:
    saved_q_table = pickle.load(file)

agent.q_table = defaultdict(
    lambda: np.zeros(agent.number_of_actions)
)
agent.q_table.update(saved_q_table)

print("Q-table loaded successfully!")
print("Number of learned states:", len(agent.q_table))


observation, info = env.reset()
state = state_discretizer.discretize(observation)

terminated = False
truncated = False
total_reward = 0
number_of_steps = 0

while not terminated and not truncated:

    action_id = agent.select_action(state)
    continuous_action = action_discretizer.get_action(action_id)

    next_observation, reward, terminated, truncated, info = env.step(continuous_action)
    state = state_discretizer.discretize(next_observation)

    total_reward += reward
    number_of_steps += 1

print("\nDemo finished!")
print(f"Total Reward: {total_reward:.2f}")
print(f"Number of Steps: {number_of_steps}")

env.close()
