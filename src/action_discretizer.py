import numpy as np

class ActionDiscretizer:
    """Converts a finite set of discrete actions IDs into continuous 6-D HalfCheetah actions"""

    def __init__(self, action_dimention=6):
        self.action_dimention = action_dimention
        self.actions = self._create_action_catalogue()

    def _create_action_catalogue(self):
        actions = []
        torque_levels = [-1.0, -0.5, 0.5, 1.0]

        # 0: No torque
        actions.append(np.zeros(self.action_dimention))

        # Apply the same torque to all the joints
        for torque in torque_levels:
            actions.append(
                    np.full(
                        self.action_dimention, torque, dtype=np.float32
                    )
            )

        # Activate one joint at a time
        for joint in range(self.action_dimention):
            for torque in torque_levels:
                action = np.zeros(
                    self.action_dimention,
                    dtype=np.float32
                )

                action[joint] = torque
                actions.append(action)

        return np.array(actions, dtype=np.float32)

    def get_action(self, action_id):
        return self.actions[action_id]

    def get_number_of_actions(self):
        return len(self.actions)

