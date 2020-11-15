from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

PLAYGROUND_SIZEX = 10
PLAYGROUND_SIZEY = 10
SCALE_FACTORX = 15
SCALE_FACTORY = 15
MAX_LENGTH_SNAKE = 40 #int(PLAYGROUND_SIZEX * PLAYGROUND_SIZEY)


@dataclass
class Point:
    x: int
    y: int

    def __add__(self, other_point):
        result = Point(0, 0)
        result.x = (PLAYGROUND_SIZEX + (
                (self.x + other_point.x) % PLAYGROUND_SIZEX)) % PLAYGROUND_SIZEX  # such that '-1 % 15 == 14'!
        result.y = (PLAYGROUND_SIZEY + ((self.y + other_point.y) % PLAYGROUND_SIZEY)) % PLAYGROUND_SIZEY
        return result

    def __sub__(self, other_point):
        result = Point(0, 0)
        result.x = (PLAYGROUND_SIZEX + (
                (self.x - other_point.x) % PLAYGROUND_SIZEX)) % PLAYGROUND_SIZEX  # such that '-1 % 15 == 14'!
        result.y = (PLAYGROUND_SIZEY + ((self.y - other_point.y) % PLAYGROUND_SIZEY)) % PLAYGROUND_SIZEY
        return result

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


@dataclass
class State:
    snake: []  # this will be a list of points
    goal: Point
    snake_length: int

    def to_array(self):
        state = np.zeros(2 * MAX_LENGTH_SNAKE + 2)
        i = 0
        for point in self.snake:
            state[i] = point.x
            state[i + 1] = point.y
            i += 2
        state[-2] = self.goal.x
        state[-1] = self.goal.y
        return state

    def to_image(self):
        state = np.zeros((PLAYGROUND_SIZEX, PLAYGROUND_SIZEY), dtype=np.float)
        head = self.snake[0]
        state[head.x, head.y] = 200
        for point in self.snake[1:]:
            state[point.x, point.y] = 200
        state[self.goal.x, self.goal.y] = 250
        return state.reshape((1, PLAYGROUND_SIZEX, PLAYGROUND_SIZEY)) / 255.
