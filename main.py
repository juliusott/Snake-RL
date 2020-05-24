from PySide2.QtWidgets import QApplication
from GameWindow import GameWindow
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    game = GameWindow()
    app.exec_()
    sys.exit(app.exit())
