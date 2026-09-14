import gymnasium as gym

def main():
    env = gym.make("HalfCheetah-v5")

    observation, info = env.reset(seed=42)

    '''
    print("Environment created successfully")
    print(f"Observation:\n{observation}\n")
    print(f"Observation shape: \n{observation.shape}\n")
    print(f"Action space: \n{env.action_space}\n")
    print(f"Action space shape: \n{env.action_space.shape}\n")
    '''

    print(f"Initial Observation: \n{observation}\n")

    action = env.action_space.sample()

    print(f"Random sample action:\n{action}\n")

    next_observation, reward, terminated, truncated, info = env.step(action)

    print(f"Next_Observation: \n{next_observation}\n")
    print(f"Reward: \n{reward}\n")
    print(f"Terminated: \n{terminated}\n")
    print(f"Truncated: \n{truncated}\n")

    env.close()

if __name__ == "__main__":
    main()

