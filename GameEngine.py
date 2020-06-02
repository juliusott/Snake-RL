from GameWindow import GameWindow
from Globals import *
import numpy as np


class GameEngine(GameWindow):
    def __init__(self):
        super(GameEngine, self).__init__()
        self.snake_length = 1
        self.deathCount = 0
        self.GameWindow = GameWindow()
        self.state = State(snake=[None] * MAX_LENGTH_SNAKE, goal=Point(0, 0), snake_length=self.snake_length,
                           goal_collision=False,
                           self_collision=False)
        self.RandomInit()

    def RandomInit(self):
        self.state.snake[0] = Point(np.random.randint(0, PLAYGROUND_SIZEX), np.random.randint(0, PLAYGROUND_SIZEY))
        self.state.goal = Point(np.random.randint(0, PLAYGROUND_SIZEX), np.random.randint(0, PLAYGROUND_SIZEY))

    def run(self):
        self.draw(self.state)
