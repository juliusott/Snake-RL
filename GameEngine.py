from GameWindow import GameWindow
from Globals import *
import numpy as np
from PySide2.QtCore import QTimer, QObject, SIGNAL
import copy

from agent import *


class GameEngine(GameWindow):
    def __init__(self, num_episodes):
        super(GameEngine, self).__init__()
        self.iteration = 0
        self.num_episodes = num_episodes
        self.agent = Agent()
        self.reward = 0
        self.timer = QTimer(self)
        self.snake_length = 1
        self.deathCount = 0
        self.state = State(snake=[], goal=Point(0, 0), snake_length=self.snake_length)
        self.new_state = State(snake=[], goal=Point(0, 0), snake_length=self.snake_length)
        self.randomInit()
        # self.testInit()

    def testInit(self):
        self.state.snake = []
        self.state.snake.append(Point(5, 5))
        self.state.goal = Point(6, 5)

    def randomInit(self):
        self.state.snake = []
        self.state.snake.append(Point(np.random.randint(0, PLAYGROUND_SIZEX), np.random.randint(0, PLAYGROUND_SIZEY)))
        self.state.goal = Point(np.random.randint(0, PLAYGROUND_SIZEX), np.random.randint(0, PLAYGROUND_SIZEY))

    def restart(self):
        self.new_state.snake = []
        self.new_state.snake.append(
            Point(np.random.randint(0, PLAYGROUND_SIZEX), np.random.randint(0, PLAYGROUND_SIZEY)))
        self.new_state.goal = Point(np.random.randint(0, PLAYGROUND_SIZEX), np.random.randint(0, PLAYGROUND_SIZEY))
        self.state.snake_length = 1

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
        self.reward = self.state.snake_length
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

        if self.new_state.snake[0] == self.new_state.goal:
            self.new_state.snake.append(self.state.snake[-1])
            self.state.snake_length += 1
            self.reward = 5 + self.state.snake_length
            self.state.goal = Point(np.random.randint(0, PLAYGROUND_SIZEX), np.random.randint(0, PLAYGROUND_SIZEY))
            print("Hurray, got an apple")
            print(self.state.snake)
        if len(self.new_state.snake) == MAX_LENGTH_SNAKE:
            self.reward = 10 + self.state.snake_length
        elif self.new_state.snake[0] in self.state.snake:
            self.reward = -5
            print("Collision!")
            self.restart()
        observation = Transition(torch.tensor([self.state.to_array()], device=self.agent.device),
                                 torch.tensor([action], device=self.agent.device),
                                 torch.tensor([self.new_state.to_array()], device=self.agent.device),
                                 torch.tensor([self.reward], device=self.agent.device))

        self.agent.step(observation)
        # copy updated state to new state
        self.state.snake = copy.deepcopy(self.new_state.snake)
        print("New state {}".format(self.state.snake))

    def start(self):
        self.timer.timeout.connect(self.run)
        self.timer.start(200)

    def run(self, training=True):
        if training:
            self.iteration += 1
            action = self.agent.get_action(torch.tensor([self.state.to_array()], device=self.agent.device))
            self.step(action=action)
            print("reward {} in iteration {}".format(self.reward, self.iteration))
            if self.iteration % self.agent.target_update == 0:
                self.agent.update_target_network()
                print("updating target network")
        self.draw(self.state)
        #self.screenshot()
