import numpy as np

class StateDiscretizer:
    """Converts selected continuous observation features into a disrete tuple."""
    def __init__(self, feature_indices=None, number_of_bins=5, lower_bound=-5.0, upper_bound=5.0):
        
        if feature_indices is None:
            feature_indices = [0, 1, 2, 8, 9, 10]

            self.feature_indices = feature_indices
            self.number_of_bins = number_of_bins
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound

    def discretize(self, observation):
        selected_features = observation[self.feature_indices]
        clipped_features = np.clip(selected_features, self.lower_bound, self.upper_bound)

        normalized = (clipped_features - self.lower_bound) / (self.upper_bound - self.lower_bound)

        discrete_values = (normalized * self.number_of_bins).astype(int)
        discrete_values = np.clip(discrete_values, 0, self.number_of_bins-1)

        return tuple(discrete_values)

