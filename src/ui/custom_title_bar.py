from PyQt5.QtWidgets import QLabel, QHBoxLayout, QWidget
from PyQt5.QtCore import Qt

class CustomTitleBar(QWidget):
    """
    CustomTitleBar is a widget that provides a custom title bar for the application.

    This class manages the appearance and functionality of the title bar, including the title label
    and any additional buttons.
    """

    def __init__(self, parent=None):
        super(CustomTitleBar, self).__init__(parent)
        self.setFixedHeight(30)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        # Add title label
        self.titleLabel = QLabel(self)
        layout.addWidget(self.titleLabel)
        self.titleLabel.setObjectName("HexAmerous")
        self.titleLabel.setText("HexAmerous")
        self.titleLabel.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.titleLabel.setFixedHeight(30)
        layout.addWidget(self.buttons())

        # Set stylesheet
        self.setStyleSheet(
            """
            QPushButton {
                border: none;
                background-color: rgba(67, 3, 81, 0.3);
                color: #f9f9f9;
            }
            QPushButton:hover {
                background-color: rgba(67, 3, 81, 0.6);;
                color: #f9f9f9;
            }
            QPushButton:pressed {
                background-color: #430351;
                color: #f9f9f9;
            }
            QLabel {
                background-color: rgba(67, 3, 81, 0.9);
                color: #f9f9f9;
                font-size: 20pt;
                font-weight: bold;
                text-align: center;
                font-family: 'Cascadia Code';

            }
        """
        )

    def buttons(self):
        # Create buttons
        pass
