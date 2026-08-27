import os
import matplotlib.pyplot as plt


def plot_episode_rewards(dataframe, output_path, title):
    os.makedirs(os.path.dirname(output_path), exist_ok=True,)
    plt.figure(figsize=(10, 5))
    plt.plot(dataframe["episode"], dataframe["episode_reward"],)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
