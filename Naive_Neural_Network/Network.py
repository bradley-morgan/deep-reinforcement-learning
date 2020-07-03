import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

class Network(nn.Module):

    def __init__(self, input_dims, output_dims, learning_rate):
        super(Network, self).__init__()

        self.layer1 = nn.Linear(in_features=input_dims, out_features=128)
        self.layer2 = nn.Linear(in_features=128, out_features=output_dims)

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

        #Send to GPU if one is present if not use CPU
        self.device = T.device('cude:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def feed_foward(self, data):
        # convert to tensor if not
        if not T.is_tensor(data):
            data = T.tensor(data).to(self.device)

        x = F.relu(self.layer1(data))
        actions = self.layer2(x)
        return actions


    def learn(self, state, action, new_state, reward, gamma):

        # refresh the gradients so we dont recursively carry over gradients from previous passes
        self.optimiser.zero_grad()

        # encode the information into tensors (data are already 1D arrays so no reshape needed)
        state = T.tensor(state).to(self.device)
        new_state = T.tensor(new_state).to(self.device)

        # need q value for current state given action a = q(s, a)
        current_q_val = self.feed_foward(state)[action]

        # need the best action the agent could have taken which is the max q val of the new state
        new_q_val = T.max(self.feed_foward(new_state)).view(1, 1)

        target = reward + gamma * new_q_val
        cost = self.loss(target, current_q_val)

        cost.backward()
        self.optimiser.step()