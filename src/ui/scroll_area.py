from PyQt5.QtWidgets import QFrame, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt

class ScrollArea(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("background-color: transparent; border: 0px;")
        self.setWidgetResizable(True)
        self.content_widget = QWidget()
        self.content_widget.setAutoFillBackground(False)
        self.content_widget_layout = QVBoxLayout()
        self.content_widget.setLayout(self.content_widget_layout)
        self.setWidget(self.content_widget)
