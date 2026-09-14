import random
from collections import deque
import numpy as np

class ReplayBuffer:
    """
    Stores past experiences and samples random batches.

    (s, a, r, s', d) = (curr_state, action, reward, next_state, done)

    Replay Buffer
         │
         ├── Experience 1
         ├── Experience 2
         ├── Experience 3
         └── Experience 4
                  │
                  ▼
           Random mini-batch
                  │
                  ▼
           Train neural network
    """
    def __init__(self, capacity=50000, seed=42):
        self.buffer = deque(maxlen=capacity)
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """
        Store one transition.
        """
        self.buffer.append(
            (
                np.array(state, dtype=np.float32), int(action), float(reward), np.array(next_state, dtype=np.float32), bool(done)
            )
        )

    def sample(self, batch_size):
        """
        Randomly sample experiences.
        """
        experiences = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*experiences) 

        return(
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)



