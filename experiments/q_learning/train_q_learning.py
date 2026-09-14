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

import numpy as np
import pandas as pd
import pickle

from src.common.environment import HalfCheetahEnvironment
from src.common.action_discretizer import ActionDiscretizer
from src.common.state_discretizer import StateDiscretizer
from src.q_learning import QLearningAgent
from utils.plotting import plot_episode_rewards
#from utils.matrics import TrainingMatrics

def main():
    number_of_episodes = 5000
    env = HalfCheetahEnvironment(seed=42)
    action_discretizer = ActionDiscretizer()
    state_discretizer = StateDiscretizer(number_of_bins=7)
    agent = QLearningAgent(number_of_actions=action_discretizer.get_number_of_actions(), number_of_episodes=number_of_episodes)
    #matrics = TrainingMatrics()

    episode_results = []

    for episode in range(number_of_episodes):
        observation, _ = env.reset()
        state = state_discretizer.discretize(observation)

        terminated = False
        truncated = False

        episode_reward = 0.0
        action_counts = np.zeros(action_discretizer.get_number_of_actions(), dtype=int)

        while not terminated and not truncated:
            action_id = agent.select_action(state)

            continuous_action = (action_discretizer.get_action(action_id))
            next_observation, reward, terminated, truncated, info = env.step(continuous_action)
            next_state = state_discretizer.discretize(next_observation)

            done = terminated or truncated

            agent.update(
                    state=state,
                    action=action_id,
                    reward=reward,
                    next_state=next_state,
                    done=done
            )

            action_counts[action_id] += 1
            state = next_state
            episode_reward += reward
            
        max_q_value = max(
                np.max(q_values)
                for q_values in agent.q_table.values()
        )
        number_of_visited_states = len(agent.q_table)
        #matrics.add_episode(episode_reward, max_q_value, number_visited_states)

        agent.decay_epsilon()
        
        episode_results.append(
            {
                "episode": episode + 1,
                "episode_reward": episode_reward,
                "epsilon": agent.epsilon,
                "visited_states": len(agent.q_table),
                "unique_actions_used": np.count_nonzero(action_counts),
                "most_used_action": int(np.argmax(action_counts)),   
            }
        )

        print(
            f"Episode {episode + 1} | "
            f"Reward: {episode_reward:.2f} | "
            f"Epsilon: {agent.epsilon:.3f} | "
            f"Visited states: {len(agent.q_table)} | "
            f"Unique action used: {np.count_nonzero(action_counts)} | "
            f"Most used action: {int(np.argmax(action_counts))}"
        )

    with open("q_table.pkl", "wb") as file:
        pickle.dump(dict(agent.q_table), file)

    env.close()

    os.makedirs("results/q_learning", exist_ok=True)
    results_dataframe = pd.DataFrame(episode_results)
    results_dataframe.to_csv("results/q_learning/training_results.csv", index=False)

    plot_episode_rewards(
            dataframe=results_dataframe,
            output_path="results/q_learning/episode_rewards.png",
            title="Q-Learning Episode vs Rewards",
    )

if __name__ == "__main__":
    main()
