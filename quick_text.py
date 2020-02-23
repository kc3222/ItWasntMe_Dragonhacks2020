import cv2
import os
from PyQt5 import QtGui
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QAction, QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, QPoint, QRect
import sys
import time

PIC_DIR = "images"
ICON_DIR = "icons"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Write a painting")
        self.setGeometry(300, 300, 800, 600)
        # self.label = QtWidgets.QLabel()
        # self.canvas = QtGui.QPixmap(800, 600)
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        # self.label.setPixmap(self.canvas)
        menubar = self.menuBar()
        # self.setCentralWidget(self.label)
        self.time = time.time()
        self.selection_start = None
        self.selection_end = None
        self.obj_name = None
        self.doodling = True
        # self.setWindowTitle("Write a painting")

        # exitAction = QAction(QtGui.QIcon('cat.png'), 'Exit', self)
        exitAction = QAction('Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.triggered.connect(self.close)

        saveAction = QAction('Save', self)
        saveAction.setShortcut('Ctrl+S')
        saveAction.setStatusTip('Save image')
        saveAction.triggered.connect(self.save_image)

        doodleAction = QAction(QtGui.QIcon(os.path.join(ICON_DIR, "pencil_icon.png")), 'doodle', self)
        doodleAction.triggered.connect(self.start_doodling)

        writeAction = QAction(QtGui.QIcon(os.path.join(ICON_DIR, "text_icon.png")), 'write', self)
        writeAction.triggered.connect(self.start_writing)


        self.statusBar()


        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)
        fileMenu.addAction(saveAction)

        toolbar = self.addToolBar('Exit')
        toolbar.addAction(exitAction)
        toolbar.addAction(doodleAction)
        toolbar.addAction(writeAction)


        # self.setGeometry(300, 300, 350, 250)
        self.show()

    def start_doodling(self):
        self.doodling = True

    def start_writing(self):
        self.doodling = False

    def save_image(self):
        # self.canvas.toImage().save("fire.png")
        filePath, _ = QFileDialog.getSaveFileName(self, "Save image", "", "PNG(*.png);;JPEG(*.jpg *.jpeg);; All Files()")
        if filePath == "":
            return
        self.image.save(filePath)

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
            f.write("-1, -1\n")
        w = self.selection_end[0] - self.selection_start[0]
        h = self.selection_end[1] - self.selection_start[1]

        self.getText()
        pic_path = get_pic_path_by_name(self.obj_name)
        if pic_path is None:
            return
        rect = QRect(QPoint(*self.selection_start),QPoint(*self.selection_end))
        self.insert_pic(pic_path, rect)
        # self.insert_pic(pic_path, self.selection_start[0], self.selection_start[1], w, h)


    def getText(self):
        text, okPressed = QInputDialog.getText(self, "Get name", "Object's name:", QLineEdit.Normal, "")
        if okPressed:
            self.obj_name = text
            print(text)

    def insert_pic(self, pic_path, rect):
        # painter = QtGui.QPainter(self.label.pixmap())
        painter = QtGui.QPainter(self.image)
        pic = QtGui.QPixmap(pic_path)
        painter.drawPixmap(rect, pic)
        # painter.drawPixmap(x,y,w,h,pic)
        painter.end()
        self.update()

    def mouseMoveEvent(self, e):
        # painter = QtGui.QPainter(self.label.pixmap())
        painter = QtGui.QPainter(self.image)
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

    def paintEvent(self, e):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

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