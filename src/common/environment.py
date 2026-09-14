import gymnasium as gym

class HalfCheetahEnvironment:
    """
    Wrapper around the HalfCheetah-v5 gymnasium environment.
    """

    def __init__(self, seed=42):
        self.env = gym.make("HalfCheetah-v5")
        self.seed = seed

    def reset(self):
        observation, info = self.env.reset(seed=self.seed)
        return observation, info

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space
