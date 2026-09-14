import gymnasium as gym


def main():
    env = gym.make("HalfCheetah-v5")
    
    observation, info = env.reset(seed=42)

    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1

        observation = next_observation

    print("Episode finished","\n")
    print(f"Number of steps: \n{step_count}\n")
    print(f"Total_Reward: \n{total_reward}\n")

    env.close()

if __name__ == "__main__":
    main()
