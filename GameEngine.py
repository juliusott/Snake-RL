from GameWindow import GameWindow
from PySide2.QtCore import QTimer, QObject, SIGNAL
import copy

from Globals import *


class GameEngine(GameWindow):
    def __init__(self, num_episodes):
        super(GameEngine, self).__init__()
        self.done = False  # see if game over
        self.timeout = 500  # connection to the gui in milliseconds
        self.distance_to_apple = 0  # store euclidean distance from snake head to apple
        self.iteration = 0
        self.num_episodes = num_episodes  # number of games we want to play
        self.reward = 0
        self.timer = QTimer(self)
        self.snake_length = 1
        self.deathCount = 0
        self.state = State(snake=[], goal=Point(0, 0), snake_length=self.snake_length)
        self.new_state = State(snake=[], goal=Point(0, 0), snake_length=self.snake_length)
        self.randomInit()
        self.start_run = 0
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

    def reset(self):
        self.new_state.snake = []
        self.new_state.snake.append(
            Point(np.random.randint(0, PLAYGROUND_SIZEX), np.random.randint(0, PLAYGROUND_SIZEY)))
        self.new_state.goal = Point(np.random.randint(0, PLAYGROUND_SIZEX), np.random.randint(0, PLAYGROUND_SIZEY))
        self.state.snake_length = 1
        self.state.snake = copy.deepcopy(self.new_state.snake)
        self.start_run = 0
        return self.state.to_image()

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
            # print(self.state.snake)
        if self.state.snake_length == MAX_LENGTH_SNAKE:
            self.reward = 10 + self.state.snake_length
            self.done = True
        elif self.new_state.snake[0] in self.state.snake:
            self.reward = -100
            # print("Collision!")
            # print(self.state.snake)
            # print("moved " + directions[action])
            self.done = True
        # copy updated state to new state
        self.state.snake = copy.deepcopy(self.new_state.snake)

        return self.state.to_image(), self.reward, self.done

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
            self.start_run = 0
        self.draw(self.state)
