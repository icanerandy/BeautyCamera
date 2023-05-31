from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class ProcessWindow(QDialog):
    # 滑块初始化信号
    slider_init_signal = pyqtSignal()

    def __init__(self, main_signal):
        super().__init__(None)

        self.ui = None
        self.slider_change_signal = main_signal

        # 滑块
        self.brightening = 0  # 美白
        self.smooth = 0       # 磨皮
        self.face = 0         # 脸型
        self.eye = 0          # 眼睛
        self.mouth = 0        # 嘴巴
        self.eyebrow = 0      # 眉毛

        # 初始化ui
        self.init_ui()
        # 绑定槽函数
        self.action_connect()

    def init_ui(self):
        # 加载由Qt Designer设计的ui文件
        self.ui = uic.loadUi('./process_UI.ui', self)

    def action_connect(self):
        # 绑定信号
        self.slider_init_signal.connect(self.init_value)

        # 美白度
        self.ui.horizontalSlider.sliderReleased.connect(self.slider_change)
        # 磨皮程度
        self.ui.horizontalSlider_2.sliderReleased.connect(self.slider_change)
        # 脸型
        self.ui.horizontalSlider_6.sliderReleased.connect(self.slider_change)
        # 眼睛
        self.ui.horizontalSlider_7.sliderReleased.connect(self.slider_change)
        # 嘴巴
        self.ui.horizontalSlider_8.sliderReleased.connect(self.slider_change)
        # 眉毛
        self.ui.horizontalSlider_9.sliderReleased.connect(self.slider_change)

    def init_value(self):
        # 美白度
        self.ui.horizontalSlider.setValue(0)
        # 磨皮程度
        self.ui.horizontalSlider_2.setValue(0)
        # 脸型
        self.ui.horizontalSlider_6.setValue(0)
        # 眼睛
        self.ui.horizontalSlider_7.setValue(0)
        # 嘴巴
        self.ui.horizontalSlider_8.setValue(0)
        # 眉毛
        self.ui.horizontalSlider_9.setValue(0)

    def slider_change(self):
        if self.brightening != self.ui.horizontalSlider.value():
            self.brightening = self.ui.horizontalSlider.value()
            print('美白程度：', self.ui.horizontalSlider.value())
            self.slider_change_signal.emit(0, self.brightening)

        elif self.smooth != self.ui.horizontalSlider_2.value():
            self.smooth = self.ui.horizontalSlider_2.value()
            print('磨皮程度：', self.ui.horizontalSlider_2.value())
            self.slider_change_signal.emit(1, self.smooth)

        elif self.face != self.ui.horizontalSlider_6.value():
            self.face = self.ui.horizontalSlider_6.value()
            print('瘦脸程度：', self.ui.horizontalSlider_6.value())
            self.slider_change_signal.emit(2, self.face)

        elif self.eye != self.ui.horizontalSlider_7.value():
            self.eye = self.ui.horizontalSlider_7.value()
            print('大眼程度：', self.ui.horizontalSlider_7.value())
            self.slider_change_signal.emit(3, self.eye)

        elif self.mouth != self.ui.horizontalSlider_8.value():
            self.mouth = self.ui.horizontalSlider_8.value()
            print('嘴巴程度：', self.ui.horizontalSlider_8.value())
            self.slider_change_signal.emit(4, self.mouth)

        elif self.eyebrow != self.ui.horizontalSlider_9.value():
            self.eyebrow = self.ui.horizontalSlider_9.value()
            print('浓眉程度：', self.ui.horizontalSlider_9.value())
            self.slider_change_signal.emit(5, self.eyebrow)
