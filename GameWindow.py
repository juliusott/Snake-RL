from PySide2.QtWidgets import QGraphicsView, QGraphicsScene
from PySide2.QtCore import QRect
from PySide2.QtGui import QBrush


class GameWindow(QGraphicsView):
    def __init__(self):
        super().__init__(QGraphicsView)
        self.brush = QBrush()
        self.mscene = QGraphicsScene()
        self.mscene.setSceneRect(x=0.0, y=0.0, w=15.0, h=15.0)
        self.setFrameRect(QRect(left=0, top=0, width=15, height=15))
        self.setFrameStyle(4)
        self.setScene(self.mscene)
