import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import MainWindow

if __name__ == '__main__':
    #获取UIC窗口操作权限
    # app = QtWidgets.QApplication(sys.argv)
    # mw = QtWidgets.QMainWindow()
    # #调自定义的界面（即刚转换的.py对象）
    # Ui = MainWindow.MainWindow() #这里也引用了一次helloworld.py文件的名字注意
    # Ui.setupUi(mw)
    # #显示窗口并释放资源
    # mw.show()
    # sys.exit(app.exec_())

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow.MainWindow()
    window.show()
    sys.exit(app.exec_())