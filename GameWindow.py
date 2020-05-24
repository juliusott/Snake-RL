import PySide2
from PySide2.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem
from PySide2.QtCore import QRect, Qt
from PySide2.QtGui import QBrush
import numpy as np
from Globals import *


class GameWindow(QGraphicsView):
    def __init__(self, parent=None):
        super(GameWindow, self).__init__(parent=parent)
        self.brush = QBrush()
        self.scene = QGraphicsScene()
        self.scene.setSceneRect(
            QRect(left=0, top=0, width=PLAYGROUND_SIZEX * SCALE_FACTORX, height=PLAYGROUND_SIZEY * SCALE_FACTORY))
        self.setFrameRect(QRect(left=0, top=0, width=15, height=15))
        self.setFrameStyle(4)
        self.setScene(self.scene)
        self.snake = [None] * MAX_LENGTH_SNAKE
        # define SnakeHead
        self.snake[0] = QGraphicsRectItem()
        self.snake[0].setRect(QRect(10, 10, SCALE_FACTORX, SCALE_FACTORY))
        self.brush.setStyle(Qt.SolidPattern)
        self.brush.setColor(Qt.darkGreen)
        self.snake[0].setBrush(self.brush)
        self.scene.addItem(self.snake[0])
        # define rest of the snake
        for i, body in enumerate(self.snake, 2):
            body = QGraphicsRectItem()
            body.setRect(QRect(0, 0, SCALE_FACTORX, SCALE_FACTORY))
            self.brush.setColor(Qt.green)
            body.setBrush(self.brush)
            body.setVisible(False)
            self.scene.addItem(body)
        # Create the graphitem for apple
        self.goal = QGraphicsRectItem()
        self.goal.setRect(QRect(0, 0, SCALE_FACTORX, SCALE_FACTORY))
        self.brush.setStyle(Qt.SolidPattern)
        self.brush.setColor(Qt.red)
        self.goal.setBrush(self.brush)
        self.scene.addItem(self.goal)

        self.state = State(snake=self.snake, goal=Point(0, 0), snake_length=1, goal_collision=False,
                           self_collision=False)
        self.show()

    def start(self):
