import pandas as pd
import matplotlib.pyplot as plt

def main():
    q_learning = pd.read_csv("results/q_learning/training_results.csv")
    dqn = pd.read_csv("results/dqn/training_results.csv")
    ddqn = pd.read_csv("results/ddqn/training_results.csv")

    plt.figure(figsize=(12, 6))
    plt.plot(q_learning["episode"], q_learning["episode_reward"], label="Q-Learning")
    plt.plot(dqn["episode"], dqn["episode_reward"], label="DQN")
    plt.plot(ddqn["episode"], ddqn["episode_reward"], label="DDQN")

    plt.xlabel("Episode")
    plt.ylabel("Episode Return")
    plt.title("Episode Return Comparison")

    plt.legend()
    plt.tight_layout()
    plt.savefig("results/algorithm_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
