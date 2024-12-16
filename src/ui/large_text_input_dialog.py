from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton
from PyQt5.QtCore import Qt

class LargeTextInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Large Text Input")
        self.resize(600, 900)
        self.text_input = QTextEdit()
        self.text_input.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.text_input.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.text_input.setStyleSheet(
            "background-color: rgba(67, 3, 81, 0.4); color: #f9f9f9; font-family 'Cascadia Code'; font-size: 12pt; font-weight: bold;"
        )
        self.send_button = QPushButton("Send")
        self.send_button.setStyleSheet(
            "background-color:rgba(67, 3, 81, 0.4); color: #f9f9f9; font-family 'Cascadia Code'; font-size: 14pt; font-weight: bold;"
        )
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.text_input)
        self.layout.addWidget(self.send_button)
        self.send_button.clicked.connect(self.send_large_text)

    def send_large_text(self):
        """
        Sends the large text input to a designated handler.

        This method is triggered when the send button is clicked. It handles the logic for processing
        and sending the text input from the dialog.
        """
        # Logic to send large text
        pass
