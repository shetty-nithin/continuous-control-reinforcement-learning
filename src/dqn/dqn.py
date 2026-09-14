import numpy as np
import torch
import torch.nn.functional as functional

from src.dqn.models import QNetwork
from src.dqn.replay_buffer import ReplayBuffer

class DQNAgent:
    """
    Deep Q-Network agent.
    """

    def __init__(
        self,
        state_dimension,
        number_of_actions,
        learning_rate=1e-3, #0.001
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        replay_capacity=50000,
        target_update_frequency=500,
        seed=42,
    ):

        self.state_dimension = state_dimension
        self.number_of_actions = number_of_actions
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.training_steps = 0

        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.random_generator = np.random.default_rng(seed)

        # Initialising Online and Target networks.
        self.online_network = QNetwork(state_dimension, number_of_actions).to(self.device)
        self.target_network = QNetwork(state_dimension, number_of_actions).to(self.device)

        self.target_network.load_state_dict(self.online_network.state_dict())
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=learning_rate,)
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity, seed=seed,)

    def select_action(self, state):
        if (self.random_generator.random() < self.epsilon):
            return int(
                self.random_generator.integers(
                    self.number_of_actions
                )
            )

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.online_network(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train_step(self):
        if (len(self.replay_buffer) < self.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q_values = (self.online_network(states).gather(1, actions))

        with torch.no_grad():
            next_q_values = (self.target_network(next_states).max(dim=1, keepdim=True)[0])
            target_q_values = rewards + (self.discount_factor * next_q_values * (1 - dones))

        loss = functional.mse_loss(current_q_values, target_q_values,)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_steps += 1

        if (self.training_steps % self.target_update_frequency == 0):
            self.target_network.load_state_dict(
                self.online_network.state_dict()
            )

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay,)
