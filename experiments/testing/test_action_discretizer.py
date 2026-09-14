import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )
)

from src.common.action_discretizer import ActionDiscretizer

def main():
    discretizer = ActionDiscretizer()

    number_of_actions = discretizer.get_number_of_actions()
    print(f"Number of discrete actions: {number_of_actions}\n")

    for action_id in range(number_of_actions):
        action = discretizer.get_action(action_id)
        print(f"Action {action_id}: {action}")

if __name__ == "__main__":
    main()
