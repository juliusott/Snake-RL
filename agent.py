import torch
import torch.nn as nn
import torch.optim as optim
from Globals import *
from collections import deque
from collections import namedtuple
import matplotlib.pyplot as plt

F = nn.functional

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 1, stride=2)
        self.max = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(32, 64, 1, stride=2)
        self.layer1 = nn.Linear(64, 32)
        self.layer2 = nn.Linear(32, 16)
        self.action = nn.Linear(16, 4)  # 4 possible actions
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        print("input shape {}".format(x.shape))
        x = F.relu(self.conv1(x))
        x = self.max(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.action(x)
        return self.softmax(x)


def possible_actions(state):
    _possible_actions = [True, True, True, True]
    for i, direct in enumerate([Point(0, -1), Point(0, 1), Point(-1, 0), Point(1, 0)]):
        if state.snake_length > 1 and state.snake[0] + direct == state.snake[1]:
            _possible_actions[i] = False
    return _possible_actions


class Agent:
    def __init__(self, batch_size=16, buffer_size=5000, gamma=0.9, eps_start=0.9, eps_decay=1000, eps_end=0.05,
                 target_update=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.policy_network = Network().double().to(self.device)
        self.target_network = Network().to(self.device)
        self.target_network.eval()
        self.memory = Memory(max_size=buffer_size, batch_size=self.batch_size)
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_end = eps_end
        self.target_update = target_update
        self.steps_done = 0
        self.optimizer = optim.Adam(self.policy_network.parameters())
        self.plot_loss = deque(maxlen=5000)
        self.line1 = None

    def step(self, observation):
        self.memory.add(observation)
        self.optimize_model()

    def get_action(self, state):  # decaying epsilon greedy strategy
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1 * self.steps_done / self.eps_decay)
        self.steps_done += 1
        all_actions = np.arange(0, 4)
        with torch.no_grad():
            out = self.policy_network(state.double())
            print("proba distro {}".format(out))
            action = np.random.choice(all_actions, p= out.cpu().numpy()[0])
        """
        if np.random.rand() < eps_threshold:
            all_actions = np.arange(0, 4)
            action = np.random.choice(all_actions[_possible_actions.cpu().numpy()[0]])
            # print("epsilon greedy action {}, threshold: {}".format(action.item(), eps_threshold))
        else:
            with torch.no_grad():
                out = torch.zeros((1, 4), dtype=torch.double, device=self.device) -100 # this is bad but works
                out[_possible_actions] = self.policy_network(state.double())[_possible_actions]
                action = out.max(1)[-1]
                # print("q values {}".format(out))
                # print("action from network ", action.item())
        """
        return action

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())
        torch.save(self.policy_network.state_dict(), "DQN_Network.pth")

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample()
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        """ToDo: Check if the next state is a final state"""
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        new_state_batch = torch.cat(batch.next_state)
        """Compute the state-action values, the model computes Q(s) and we select the best action"""
        state_action_values = self.policy_network(state_batch).gather(1, action_batch.unsqueeze(-1))
        """next state values computed by the old target network"""
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_network(non_final_next_states.float()).max(1)[0]
        # print("next state values : {}".format(next_state_values))
        """Compute the expected Q values"""
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = F.smooth_l1_loss(state_action_values.squeeze(1), expected_state_action_values)
        # print( "Loss {}".format(loss.item()))
        self.plot_loss.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # self.live_plot()

    def plot(self):
        x = np.arange(len(self.plot_loss))
        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        self.line1, = ax.plot(x, self.plot_loss, '-o', alpha=0.8)
        plt.ylabel('Value')
        plt.title("Loss Plot")
        plt.show(block=False)



class Memory:
    def __init__(self, max_size, batch_size):
        self.buffer = deque(maxlen=max_size)
        self.batch_size = batch_size

    def add(self, x: Transition):
        # add transitions of the form: ('Transition', ('state', 'action', 'next_state', 'reward'))
        self.buffer.append(x)

    def sample(self):
        buffer_size = len(self.buffer)
        indices = np.random.choice(np.arange(buffer_size), size=self.batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)
