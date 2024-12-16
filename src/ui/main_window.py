from PyQt5.QtWidgets import QVBoxLayout, QSizeGrip
from PyQt5.QtCore import Qt
from custom_title_bar import CustomTitleBar
from chat_widget import ChatWidget

class MainWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.resize(728, 1024)
        self.flags = Qt.WindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self.setWindowFlags(self.flags)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.size_grip = QSizeGrip(self)
        self.size_grip.setStyleSheet("width: 10px; height: 5px; margin 0px;")
        self.layout.addWidget(self.size_grip)
        self.titleBar = CustomTitleBar(self)
        self.layout.addWidget(self.titleBar)
        self.chat_widget = ChatWidget()
        self.layout.addWidget(self.chat_widget)

        self.image = ""
        self.background = self.change_background_image()

    def change_background_image(self, image="./imgs/00004.png"):
        # Change background image logic
        pass
