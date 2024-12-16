from PyQt5.QtCore import QObject, pyqtSignal

class Signal(QObject):
    close_signal = pyqtSignal(name="close_signal")

    def emit(self):
        self.close_signal.emit()
