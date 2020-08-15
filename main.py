from PySide2.QtWidgets import QApplication
from GameWindow import GameWindow
from GameEngine import GameEngine
import sys
from agent import DQN

if __name__ == "__main__":
    app = QApplication(sys.argv)
    engine = GameEngine()
    engine.start()
    app.exec_()
    sys.exit(app.exit())
