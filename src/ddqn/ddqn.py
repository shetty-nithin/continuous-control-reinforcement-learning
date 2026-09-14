import torch
import torch.nn.functional as functional

from src.dqn import DQNAgent


class DDQNAgent(DQNAgent):
    """
    Double Deep Q-Network agent.
    """

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
            # Online network selects the action
            next_actions = (self.online_network(next_states).argmax(dim=1, keepdim=True))

            # Target network evaluates that action
            next_q_values = (self.target_network(next_states).gather(1, next_actions))

            target_q_values = rewards + (self.discount_factor * next_q_values * (1 - dones))

        loss = functional.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_steps += 1

        if (self.training_steps % self.target_update_frequency == 0):
            self.target_network.load_state_dict(
                self.online_network.state_dict()
            )

        return loss.item()
