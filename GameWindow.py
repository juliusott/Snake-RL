import PySide2
from PySide2.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem
from PySide2.QtCore import QRect, Qt
from PySide2.QtGui import QBrush, QPixmap
import matplotlib.pyplot as plt
from Globals import *

brushHead = QBrush()
brushHead.setStyle(Qt.SolidPattern)
brushHead.setColor(Qt.darkGreen)

brushBody = QBrush()
brushBody.setStyle(Qt.SolidPattern)
brushBody.setColor(Qt.green)

brushGoal = QBrush()
brushGoal.setStyle(Qt.SolidPattern)
brushGoal.setColor(Qt.red)


class GameWindow(QGraphicsView):
    def __init__(self, parent=None):
        super(GameWindow, self).__init__(parent=parent)
        self.screen = None
        self.scene = QGraphicsScene()
        self.scene.setSceneRect(
            QRect(left=0, top=0, width=PLAYGROUND_SIZEX * SCALE_FACTORX, height=PLAYGROUND_SIZEY * SCALE_FACTORY))
        self.setFrameStyle(4)
        self.setScene(self.scene)
        self.snake = []
        # define SnakeHead
        self.snake.append(QGraphicsRectItem())
        self.snake[0].setRect(QRect(0, 0, SCALE_FACTORX, SCALE_FACTORY))

        self.snake[0].setBrush(brushHead)
        self.scene.addItem(self.snake[0])
        # define rest of the snake
        for i in range(1, MAX_LENGTH_SNAKE):
            self.snake.append(QGraphicsRectItem())
            self.snake[i].setRect(QRect(0, 0, SCALE_FACTORX, SCALE_FACTORY))
            self.snake[i].setBrush(brushBody)
            self.snake[i].setVisible(False)
            self.scene.addItem(self.snake[i])
        # Create the graphic item for apple
        self.goal = QGraphicsRectItem()
        self.goal.setRect(QRect(0, 0, SCALE_FACTORX, SCALE_FACTORY))
        self.goal.setBrush(brushGoal)
        self.scene.addItem(self.goal)

        self.show()

    def draw(self, state):
        self.goal.setPos(state.goal.x * SCALE_FACTORX, state.goal.y * SCALE_FACTORY)
        self.goal.setVisible(True)
        self.snake[0].setPos(float(state.snake[0].x * SCALE_FACTORX), float(state.snake[0].y * SCALE_FACTORY))
        for i in range(1, state.snake_length):
            self.snake[i].setPos(state.snake[i].x * SCALE_FACTORX, state.snake[i].y * SCALE_FACTORY)
            self.snake[i].setVisible(True)
        for i in range(state.snake_length, MAX_LENGTH_SNAKE):
            self.snake[i].setVisible(False)

    def screenshot(self):
        self.screen = self.grab()
        self.screen.save("./snake_screen.png")
