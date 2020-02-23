import cv2
import os
from PyQt5 import QtGui
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QPoint, QRect
import sys
import time

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
        self.time = time.time()
        self.selection_start = None
        self.selection_end = None
        self.obj_name = None
        #self.draw_something()

    def mousePressEvent(self, e):
        # if event.buttons() & QtCore.Qt.LeftButton:
        self.selection_start = (e.x(), e.y())

    def mouseReleaseEvent(self, e):
        self.selection_end = (e.x(), e.y())
        # painter = QtGui.QPainter(self.label.pixmap())
        # pen = QtGui.QPen()
        # pen.setWidth(10)
        # pen.setColor(QtGui.QColor('red'))
        # painter.setPen(pen)
        with open("out.csv", "a") as f:
            f.write("\n")
        w = self.selection_end[0] - self.selection_start[0]
        h = self.selection_end[1] - self.selection_start[1]

        self.getText()
        pic_path = get_pic_path_by_name(self.obj_name)
        if pic_path is None:
            return
        rect = QRect(QPoint(*self.selection_start),QPoint(*self.selection_end))
        self.insert_pic(pic_path, rect)
        # self.insert_pic(pic_path, self.selection_start[0], self.selection_start[1], w, h)


    def draw_something(self):
        painter = QtGui.QPainter(self.label.pixmap())
        painter.drawLine(10, 10, 300, 200)
        painter.end()

    def getText(self):
        text, okPressed = QInputDialog.getText(self, "Get name", "Object's name:", QLineEdit.Normal, "")
        if okPressed:
            self.obj_name = text
            print(text)

    def insert_pic(self, pic_path, rect):
        painter = QtGui.QPainter(self.label.pixmap())
        pic = QtGui.QPixmap(pic_path)
        painter.drawPixmap(rect, pic)
        # painter.drawPixmap(x,y,w,h,pic)
        painter.end()
        self.update()

    def mouseMoveEvent(self, e):
        painter = QtGui.QPainter(self.label.pixmap())
        pen = QtGui.QPen()
        pen.setWidth(10)
        pen.setColor(QtGui.QColor('red'))
        painter.setPen(pen)
        painter.drawPoint(e.x(), e.y(), )
        if time.time() - self.time > 0.2:
            self.time = time.time()
            with open("out.csv", "a") as f:
                f.write(f"{e.x()}, {e.y()}\n")
        painter.end()
        self.update()

def get_pic_by_name(name):
    name_dir = os.path.join(PIC_DIR, name)
    file = os.listdir(name_dir)[0]
    img = cv2.imread(os.path.join(name_dir,file))
    return img

def get_pic_path_by_name(name):
    name_dir = os.path.join(PIC_DIR, name)
    if not os.path.isdir(name_dir):
        print(f"{name} is not yet supported")
        return
    file = os.listdir(name_dir)[0]
    return os.path.join(name_dir,file)

if __name__ == "__main__":
    # img = get_pic_by_name("dog")
    # print(img.shape)
    # cv2.imshow("Image", img)
    # cv2.waitKey(1)
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()