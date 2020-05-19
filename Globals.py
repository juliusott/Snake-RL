from dataclasses import dataclass
@dataclass
class Point:
    x: int
    y: int
    def __add__(self, other_point):
        result = Point();
        result.x = self.x + other_point.x
        result.y = self.y + other_point.y
        return result

    def __sub__(self, other_point):
        result = Point();
        result.x = self.x - other_point.x
        result.y = self.y - other_point.y
        return result
@dataclass
class State:
    snake: [] #this will be a list of points
    goal: Point
    snake_length: int
    goal_collision: bool
    self_collision: bool

