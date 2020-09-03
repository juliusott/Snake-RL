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
        self.layer1 = nn.Linear(2 * MAX_LENGTH_SNAKE + 2, 512)
        self.drop = nn.Dropout(0.5)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 64)
        self.action = nn.Linear(64, 4)  # 4 possible actions

    def forward(self, x):
        x = F.relu(self.drop(self.layer1(x)))
        x = F.relu(self.drop(self.layer2(x)))
        x = F.relu(self.drop(self.layer3(x)))
        x = F.relu(self.drop(self.layer4(x)))
        x = self.action(x)
        return x


class Agent:
    def __init__(self, batch_size=512, buffer_size=5000, gamma=0.9, eps_start=0.9, eps_decay=200, eps_end=0.05,
                 target_update=20):
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
        self.optimizer = optim.RMSprop(self.policy_network.parameters())
        self.plot_buffer = []
        self.plot_means = deque(maxlen=100)
        self.line1 = None

    def step(self, observation):
        self.memory.add(observation)
        self.optimize_model()

    def get_action(self, state):  # decaying epsilon greedy strategy
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1 * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if np.random.rand() < eps_threshold:
            action = torch.randint(0, 3, (1,))
            # print("epsilon greedy action {}, threshold: {}".format(action.item(), eps_threshold))
        else:
            with torch.no_grad():
                action = self.policy_network(state.double()).max(1)[1]
                # print("action from network ", action.item())
        return action.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample()
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        """ToDo: Check if the next state is a final state"""
        new_state_batch = torch.cat(batch.next_state)
        """Compute the state-action values, the model computes Q(s) and we select the best action"""
        state_action_values = self.policy_network(state_batch).gather(1, action_batch.unsqueeze(-1))
        """next state values computed by the old target network"""
        next_state_values = self.target_network(new_state_batch.float()).max(1)[0]
        """Compute the expected Q values"""
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = F.smooth_l1_loss(state_action_values.squeeze(1), expected_state_action_values)
        # print( "Loss {}".format(loss.item()))
        self.optimizer.zero_grad()
        loss.backward()
        """TODO: Add gradient clipping here"""
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        if len(self.plot_buffer) < 100:
            self.plot_buffer.append(np.around(loss.item(), decimals=3))
            print("add loss to buffer {}".format(loss.item()))
            print("buffer length {}".format(len(self.plot_buffer)))
        else:
            mean = np.mean(self.plot_buffer)
            self.plot_means.append(np.around(mean, decimals=3))
            print("mean loss after 100 iterations {}".format(self.plot_means))
            self.plot_buffer = []
        if len(self.plot_means) > 10:
            self.live_plot()

    def live_plot(self, pause_time=0.1):
        x = np.arange(len(self.plot_means))
        if not self.line1:
            # this is the call to matplotlib that allows dynamic plotting
            print("Creating new plot")
            plt.ion()
            fig = plt.figure(figsize=(13, 6))
            ax = fig.add_subplot(111)
            # create a variable for the line so we can later update it
            self.line1, = ax.plot(x, self.plot_means, '-o', alpha=0.8)
            # update plot label/title
            plt.ylabel('Value')
            plt.title("Loss Plot")
            plt.show()

        # after the figure, axis, and line are created, we only need to update the y-data
        self.line1.set_data(x, self.plot_means)
        # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
        plt.pause(pause_time)


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
