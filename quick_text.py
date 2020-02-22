import cv2
import os
from PyQt5 import QtGui
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt
import sys

PIC_DIR = "images"

'''
I need to draw a rect
Click and drag
Then get user input from openCV. Can I do so?
'''

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.label = QtWidgets.QLabel()
        canvas = QtGui.QPixmap(800, 600)
        self.label.setPixmap(canvas)
        self.setCentralWidget(self.label)
        #self.draw_something()

    def draw_something(self):
        painter = QtGui.QPainter(self.label.pixmap())
        painter.drawLine(10, 10, 300, 200)
        painter.end()

    def mouseMoveEvent(self, e):
        painter = QtGui.QPainter(self.label.pixmap())
        pen = QtGui.QPen()
        pen.setWidth(10)
        pen.setColor(QtGui.QColor('red'))
        painter.setPen(pen)
        painter.drawPoint(e.x(), e.y(), )
        painter.end()
        self.update()

def get_pic_by_name(name):
    name_dir = os.path.join(PIC_DIR, name)
    file = os.listdir(name_dir)[0]
    img = cv2.imread(os.path.join(name_dir,file))
    return img


if __name__ == "__main__":
    # img = get_pic_by_name("dog")
    # print(img.shape)
    # cv2.imshow("Image", img)
    # cv2.waitKey(1)
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()