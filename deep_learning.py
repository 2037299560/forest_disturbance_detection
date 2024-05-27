from PyQt5.QtWidgets import QFileDialog, QPushButton


def load_file(self):
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly
    file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                               "All Files (*);;Python Files (*.py)", options=options)
    if file_name:
        print(file_name)


