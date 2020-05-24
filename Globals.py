from dataclasses import dataclass
from PySide2.QtWidgets import QGraphicsRectItem
PLAYGROUND_SIZEX = 100
PLAYGROUND_SIZEY = 100
SCALE_FACTORX = 10
SCALE_FACTORY = 10
MAX_LENGTH_SNAKE = int((PLAYGROUND_SIZEX * PLAYGROUND_SIZEY)/(SCALE_FACTORX*SCALE_FACTORY))
@dataclass
class Point:
    x: int
    y: int

    def __add__(self, other_point):
        result = Point()
        result.x = (WINDOW_SIZEX + ((self.x + other_point.x) % WINDOW_SIZEX)) % WINDOW_SIZEX  # Damit '-1 % 15 == 14'!
        result.y = (WINDOW_SIZEY + ((self.y + other_point.y) % WINDOW_SIZEY)) % WINDOW_SIZEY
        return result

    def __sub__(self, other_point):
        result = Point()
        result.x = (WINDOW_SIZEX + ((self.x - other_point.x) % WINDOW_SIZEX)) % WINDOW_SIZEX  # Damit '-1 % 15 == 14'!
        result.y = (WINDOW_SIZEY + ((self.y - other_point.y) % WINDOW_SIZEY)) % WINDOW_SIZEY
        return result


@dataclass
class State:
    snake: []  # this will be a list of points
    goal: Point
    snake_length: int
    goal_collision: bool
    self_collision: bool
