import cv2
from PyQt5 import QtWidgets  # import PyQt5 widgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap
import sys

import GUI


class MainWindow:
    def __init__(self):
        app = QtWidgets.QApplication(sys.argv)
        window = QtWidgets.QMainWindow()
        self.image = None
        self.res_image = None
        self.ui = GUI.Ui_MainWindow()
        self.ui.setupUi(window)
        self.action_connect()
        window.show()
        sys.exit(app.exec_())

    def action_connect(self):
        self.ui.action_O.triggered.connect(self.load_image)
        self.ui.action_S.triggered.connect(self.save_image)

    def load_image(self):
        # 打开单个文件对话框
        # 下行代码第三个参数是默认路径，用 "."代替当前
        # 第四个参数：'图片文件 (*.jpg)'改成选中两种类型时有问题 '图片文件 (*.png, *.jpg)'
        # 弹出来的显示图片的窗口会随着图片尺寸大小的变化而变化
        img_name, _ = QFileDialog.getOpenFileName(None, '打开文件', '.', '图片文件 (*.png, *.jpg)')
        # 得到图片文件名
        self.image = img_name
        self.res_image = img_name
        self.show_image()

    def save_image(self):
        img_name, _ = QFileDialog.getSaveFileName(None, '打开文件', '.', '图片文件 (*.png, *.jpg)')
        cv2.imwrite(img_name, self.res_image)

    def show_image(self):
        self.ui.label.setPixmap(QPixmap(self.image))


if __name__ == "__main__":
    MainWindow()
