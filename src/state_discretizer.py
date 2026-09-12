import numpy as np

class StateDiscretizer:
    """Converts selected continuous observation features into a disrete tuple."""
    def __init__(self, feature_indices=None, number_of_bins=5, lower_bound=None, upper_bound=None):
        
        if feature_indices is None:
            feature_indices = [0, 1, 8] #indices of torso height, torso angle, forward (x) velocity

        self.feature_indices = feature_indices
        self.number_of_bins = number_of_bins

        if lower_bound is None:
            lower_bound = np.array([-0.5, -0.5, -5.0])   # height, angle, velocity
        if upper_bound is None:
            upper_bound = np.array([0.5, 0.5, 5.0])

        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)

    def discretize(self, observation):
        selected_features = observation[self.feature_indices]
        clipped_features = np.clip(selected_features, self.lower_bound, self.upper_bound)

        normalized = (clipped_features - self.lower_bound) / (self.upper_bound - self.lower_bound)

        discrete_values = (normalized * self.number_of_bins).astype(int)
        discrete_values = np.clip(discrete_values, 0, self.number_of_bins-1)

        return tuple(discrete_values)

