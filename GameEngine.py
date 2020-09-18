from GameWindow import GameWindow
import sys
import numpy as np
from PySide2.QtCore import QTimer, QObject, SIGNAL
import copy

from agent import *


class GameEngine(GameWindow):
    def __init__(self, num_episodes):
        super(GameEngine, self).__init__()
        self.done = False  # see if game over
        self.timeout = 15  # connection to the gui in milliseconds
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
        self.distance_to_apple = self.distance()
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
            self.reward = 10
            self.state.goal = Point(np.random.randint(0, PLAYGROUND_SIZEX), np.random.randint(0, PLAYGROUND_SIZEY))
            print("Hurray, got an apple")
            print(self.state.snake)
        if len(self.new_state.snake) == MAX_LENGTH_SNAKE:
            self.reward = 10 + self.state.snake_length
        elif self.new_state.snake[0] in self.state.snake:
            self.reward = -100
            print("Collision!")
            print(self.state.snake)
            print("moved " + directions[action])
            self.done = True
        elif self.state.snake_length == MAX_LENGTH_SNAKE:
            self.done = True

        # copy updated state to new state
        self.state.snake = copy.deepcopy(self.new_state.snake)
        if self.distance_to_apple > self.distance():
            self.reward = 1
        else:
            self.reward = -1
        # print("New state {}".format(self.state.snake))

    def start(self):
        self.timer.timeout.connect(self.run)
        self.timer.start(self.timeout)

    def distance(self):
        """computes the euclidean distance between th snake head and apple"""
        snake_head = self.state.snake[0]
        return -np.sqrt((snake_head.x - self.state.goal.x) ** 2 + (snake_head.y - self.state.goal.y) ** 2)

    def run(self, training=True):
        if training:
            for i_episode in range(self.num_episodes):
                self.iteration = 0
                last_screen = self.state.to_image()
                current_screen = self.state.to_image()
                state = current_screen - last_screen
                self.done = False
                while not self.done:
                    self.iteration += 1
                    pos_a = possible_actions(self.state)
                    action = self.agent.get_action(torch.tensor([state], device=self.agent.device),
                                                   torch.tensor([pos_a], device=self.agent.device))

                    last_screen = current_screen
                    current_screen = self.state.to_image()
                    if not self.done:
                        next_state = current_screen - last_screen
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
                    if self.done:
                        self.restart()
                if i_episode % self.agent.target_update == 0:
                    self.agent.update_target_network()
                    print("updating target network")
            training = False
        else:
            self.timeout = 500
            self.agent.policy_network.eval()
            pos_a = possible_actions(self.state)
            action = self.agent.get_action(torch.tensor([self.state.to_image()], device=self.agent.device),
                                           torch.tensor([pos_a], device=self.agent.device))
            self.step(action=action)
            self.draw(self.state)
