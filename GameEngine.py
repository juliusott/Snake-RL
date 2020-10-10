from GameWindow import GameWindow
import sys
import numpy as np
from PySide2.QtCore import QTimer, QObject, SIGNAL
import copy
from itertools import count
from agent import *


class GameEngine(GameWindow):
    def __init__(self, num_episodes):
        super(GameEngine, self).__init__()
        self.done = False  # see if game over
        self.timeout = 500  # connection to the gui in milliseconds
        self.distance_to_apple = 0  # store euclidean distance from snake head to apple
        self.iteration = 0
        self.num_episodes = num_episodes  # number of games we want to play
        self.agent = Agent()
        self.reward = 0
        self.timer = QTimer(self)
        self.snake_length = 1
        self.deathCount = 0
        self.state = State(snake=[], goal=Point(0, 0), snake_length=self.snake_length)
        self.new_state = State(snake=[], goal=Point(0, 0), snake_length=self.snake_length)
        self.randomInit()
        self.start_run = 0
        self.network_state = None
        self.screens = deque(maxlen=4)  # maxlen determines how many screens you want to give to the agent
        self.high_score = 0

    def testInit(self):
        self.state.snake = []
        self.state.snake.append(Point(5, 5))
        self.state.goal = Point(6, 5)

    def randomInit(self):
        self.start_run = 0
        self.state.snake = []
        self.state.snake.append(Point(np.random.randint(0, PLAYGROUND_SIZEX), np.random.randint(0, PLAYGROUND_SIZEY)))
        self.state.goal = Point(np.random.randint(0, PLAYGROUND_SIZEX), np.random.randint(0, PLAYGROUND_SIZEY))

    def restart(self):
        self.new_state.snake = []
        self.new_state.snake.append(
            Point(np.random.randint(0, PLAYGROUND_SIZEX), np.random.randint(0, PLAYGROUND_SIZEY)))
        self.new_state.goal = Point(np.random.randint(0, PLAYGROUND_SIZEX), np.random.randint(0, PLAYGROUND_SIZEY))
        self.state.snake_length = 1
        self.state.snake = copy.deepcopy(self.new_state.snake)
        self.start_run = 0

    def action_to_one_hot(self, action):
        action_one_hot = np.zeros(4)
        action_one_hot[action] = 1
        return action_one_hot

    def step(self, action):
        directions = ["up", "down", "left", "right"]
        """
        always look in point of the compass
        up = North
        down = South
        right = East
        left = West
        """
        self.distance_to_apple = self.distance(self.state)
        # print("Current state snake {}".format(self.state.snake))
        self.new_state = copy.deepcopy(self.state)
        if directions[action] == "up":
            self.new_state.snake[0] = self.state.snake[0] + Point(0, -1)
        elif directions[action] == "down":
            self.new_state.snake[0] = self.state.snake[0] + Point(0, 1)
        elif directions[action] == "left":
            self.new_state.snake[0] = self.state.snake[0] + Point(-1, 0)
        elif directions[action] == "right":
            self.new_state.snake[0] = self.state.snake[0] + Point(1, 0)

        # print("snake length is {}, should {} ".format(new_state.snake_length, self.state.snake_length))
        for i in range(self.state.snake_length - 1):
            self.new_state.snake[i + 1] = copy.copy(self.state.snake[i])

        if self.distance_to_apple > self.distance(self.new_state):
            self.reward = 1
        else:
            self.reward = -1
        if self.new_state.snake[0] == self.new_state.goal:
            self.new_state.snake.append(self.state.snake[-1])
            self.state.snake_length += 1
            self.reward = 10
            self.state.goal = Point(np.random.randint(0, PLAYGROUND_SIZEX), np.random.randint(0, PLAYGROUND_SIZEY))
            print("Hurray, got an apple")
            print(self.state.snake)
        if self.state.snake_length == MAX_LENGTH_SNAKE:
            self.reward = 10 + self.state.snake_length
            self.done = True
        elif self.new_state.snake[0] in self.state.snake:
            self.reward = -100
            print("Collision!")
            print(self.state.snake)
            print("moved " + directions[action])
            self.done = True
        # copy updated state to new state
        self.state.snake = copy.deepcopy(self.new_state.snake)

        # print("New state {}".format(self.state.snake))

    def start(self):
        self.timer.timeout.connect(self.run)
        self.timer.start(self.timeout)

    def distance(self, state):
        """computes the manhattan distance between the snake head and apple
        Note: this distance works only for a squared playground"""
        snake_head = state.snake[0]
        dis_x = np.abs(snake_head.x - self.state.goal.x) % (PLAYGROUND_SIZEX - 1)
        dis_y = np.abs(snake_head.y - self.state.goal.y) % (PLAYGROUND_SIZEX - 1)
        return dis_x + dis_y

    def training(self):
        for i_episode in range(self.num_episodes):
            self.iteration = 0
            self.screens.append(self.state.to_image())
            self.screens.append(self.state.to_image())
            self.screens.append(self.state.to_image())
            self.screens.append(self.state.to_image())
            state = np.squeeze(self.screens)
            print(np.shape(state))
            self.done = False
            while not self.done:
                self.iteration += 1
                action = self.agent.get_action(torch.tensor([state], device=self.agent.device))


                # print("allowed actions {} and action {}".format(pos_a, action))
                self.screens.append(self.state.to_image())
                if not self.done:
                    next_state =  np.squeeze(self.screens)
                else:
                    next_state = None

                self.step(action=action)
                observation = Transition(
                    torch.tensor([state], device=self.agent.device),
                    torch.tensor([action], device=self.agent.device),
                    torch.tensor([next_state], device=self.agent.device),
                    torch.tensor([self.reward], device=self.agent.device, dtype=torch.double))

                self.agent.step(observation)

                state = next_state
                print("reward {} in epsiode {}".format(self.reward, i_episode))
                if self.state.snake_length > self.high_score:
                    self.high_score = self.state.snake_length
                    print("new high score {}".format(self.high_score))
                if self.done:
                    self.restart()
            if i_episode % self.agent.target_update == 0:
                self.agent.update_target_network()
                print("updating target network")
        print("high score {}".format(self.high_score))
        self.agent.plot()

    def run(self):
        if self.start_run == 0:
            self.screens.append(self.state.to_image())
            self.screens.append(self.state.to_image())
            self.screens.append(self.state.to_image())
            self.screens.append(self.state.to_image())
            self.network_state = np.squeeze(self.screens)
        self.start_run = 1
        self.done = False
        self.timeout = 500
        self.agent.policy_network.eval()
        action = self.agent.get_action(torch.tensor([self.network_state], device=self.agent.device),
                                       )
        self.screens.append(self.state.to_image())
        if not self.done:
            next_state = np.squeeze(self.screens)
        else:
            next_state = None
        self.network_state = next_state
        self.step(action=action)
        if self.done:
            self.restart()
        self.draw(self.state)
