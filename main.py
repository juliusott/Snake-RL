from PySide2.QtWidgets import QApplication
from GameWindow import GameWindow
from GameEngine import GameEngine
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    engine = GameEngine(num_episodes=20)
    engine.training()
    engine.start()
    app.exec_()
    sys.exit(app.exit())
