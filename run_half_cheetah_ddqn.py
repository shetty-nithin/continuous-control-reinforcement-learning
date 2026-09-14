import gymnasium as gym
import torch

from src.common.action_discretizer import ActionDiscretizer
from src.dqn.models import QNetwork

env = gym.make("HalfCheetah-v5", render_mode="human")
action_discretizer = ActionDiscretizer()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

state_dimension = env.observation_space.shape[0]
number_of_actions = action_discretizer.get_number_of_actions()

online_network = QNetwork(state_dimension, number_of_actions).to(device)
online_network.load_state_dict(torch.load("results/ddqn/ddqn_model.pt", map_location=device))
online_network.eval()

print("DDQN model loaded successfully!")

observation, info = env.reset()
terminated = False
truncated = False
total_reward = 0
number_of_steps = 0

while not terminated and not truncated:
    state_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        q_values = online_network(state_tensor)

    action_id = int(torch.argmax(q_values, dim=1).item())
    continuous_action = action_discretizer.get_action(action_id)

    next_observation, reward, terminated, truncated, info = env.step(continuous_action)
    observation = next_observation

    total_reward += reward
    number_of_steps += 1

print("\nDemo finished!")
print(f"Total Reward: {total_reward:.2f}")
print(f"Number of Steps: {number_of_steps}")

env.close()
