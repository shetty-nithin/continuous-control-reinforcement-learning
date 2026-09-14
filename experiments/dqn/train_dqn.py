import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )
)

import pandas as pd
import numpy as np
import torch

from src.common.environment import HalfCheetahEnvironment
from src.common.action_discretizer import ActionDiscretizer
from src.dqn import DQNAgent


def main():
    number_of_episodes = 100
    env = HalfCheetahEnvironment(seed=42)
    action_discretizer = ActionDiscretizer()
    state_dimension = env.observation_space.shape[0]

    number_of_actions = action_discretizer.get_number_of_actions()

    agent = DQNAgent(
        state_dimension=state_dimension,
        number_of_actions=number_of_actions,
    )

    episode_results = []

    for episode in range(number_of_episodes):
        state, _ = env.reset()

        terminated = False
        truncated = False

        episode_reward = 0.0
        episode_losses = []

        action_counts = np.zeros(number_of_actions, dtype=int)

        while not terminated and not truncated:
            action_id = agent.select_action(state)
            continuous_action = action_discretizer.get_action(action_id)

            next_state, reward, terminated, truncated, info = env.step(continuous_action)
            done = terminated or truncated

            agent.store_transition(state, action_id, reward, next_state, done)

            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)

            action_counts[action_id] += 1
            state = next_state
            episode_reward += reward

        agent.decay_epsilon()

        mean_loss = (
            float(np.mean(episode_losses))
            if episode_losses
            else np.nan
        )

        episode_results.append(
            {
                "episode": episode + 1,
                "episode_reward": episode_reward,
                "mean_loss": mean_loss,
                "epsilon": agent.epsilon,
            }
        )

        print(
            f"Episode {episode + 1} | "
            f"Reward: {episode_reward:.2f} | "
            f"Loss: {mean_loss:.6f} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

    env.close()

    os.makedirs("results/dqn", exist_ok=True)
    torch.save(agent.online_network.state_dict(), "results/dqn/dqn_model.pt")

    dataframe = pd.DataFrame(episode_results)
    dataframe.to_csv("results/dqn/training_results.csv", index=False)

if __name__ == "__main__":
    main()
