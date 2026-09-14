import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """
    Neural Network that creates a Q-value for every discrete action.
    """
    def __init__(self, state_dimention, number_of_actions):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dimention, 128), # Input_Layer -> Hidden_Layer_1
            nn.ReLU(),                       # Non-linear activation
            nn.Linear(128, 128),             # Hidden_Layer_1 -> Hidden_Layer_2
            nn.ReLU(),                       # Non-linear activation
            nn.Linear(128, number_of_actions)# Hidden_Layer_2 -> Output_Layer
        )

    def forward(self, state):
        """
        Forward pass through neural network.
        """
        return self.network(state)
