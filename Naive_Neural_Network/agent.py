# learn
# constructor *
# decay epsilon*
# choose_action*

import numpy as np
from Network import Network


class Agent():

    def __init__(self, state_dims, action_dims, epsilon, min_epsilon, epsilon_decay, gamma, learning_rate):

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_rate = epsilon_decay

        self.state_dims = state_dims
        self.action_dims = action_dims

        self.gamma = gamma
        self.learning_rate = learning_rate

        self.network = Network(input_dims=state_dims, output_dims=action_dims, learning_rate=learning_rate)

    def decay_epsilon(self):

        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon * self.epsilon_decay_rate

        else:
            self.epsilon = self.min_epsilon

    def choose_action(self, state):

        if np.random.random() > self.epsilon:
            # exploit environemnt
            actions = self.network.feed_foward(state)
            _, max_act_ind = actions.max(0)
            action = int(max_act_ind.item())
            return action

        else:
            action = np.random.randint(low=0, high=self.action_dims)
            return action

    def learn(self,state, action, new_state, reward ):
        self.network.learn(state, action, new_state, reward, self.gamma)
        self.decay_epsilon()


if __name__ == "__main__":
    pass