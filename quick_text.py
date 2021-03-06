import cv2
import os
from PyQt5 import QtGui
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QAction, QFileDialog, QListWidget, QListWidgetItem, QDialog, QComboBox
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, QPoint, QRect
import sys
import time
import sys
sys.path.insert(1, './fast-style-transfer-master')
sys.path.insert(2, './fast-style-transfer-master/src')
from evaluate import ffwd_to_img
from combine_quickdraw import combine_quickdraw
from find_bounding_box import find_bounding_box
import random
import shutil

PIC_DIR = "images"
ICON_DIR = "icons"
STROKE_DIR = "strokes"
STROKE_FILE = "out.csv"
TEMP_DIR = "temp"
OUTPUT_DIR = "outputs"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.id = 0
        self.setWindowTitle("Write A Painting")
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
        # self.temp_file = os.path.join(TEMP_DIR, "temp.jpg")
        self.image.save(self.get_temp_file())
        shutil.copyfile(STROKE_FILE, os.path.join(STROKE_DIR, f"out_{self.id}.csv")) 
        self.types = self.get_available_objects()
        self.start_doodle_id = self.id
        

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

        classifyAction = QAction(QtGui.QIcon(os.path.join(ICON_DIR, "star_icon.png")), 'doodle to real image', self)
        classifyAction.triggered.connect(self.classify)

        beautifyAction = QAction(QtGui.QIcon(os.path.join(ICON_DIR, "mona_icon.png")), 'turn into artwork', self)
        beautifyAction.triggered.connect(self.beautify)

        clearAction = QAction(QtGui.QIcon(os.path.join(ICON_DIR, "new_icon.png")), 'blank canvas', self)
        clearAction.triggered.connect(self.clear)

        undoAction = QAction(QtGui.QIcon(os.path.join(ICON_DIR, "undo_icon.png")), 'undo', self)
        undoAction.setShortcut('Ctrl+Z')
        undoAction.triggered.connect(self.undo)

        self.statusBar()

        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)
        fileMenu.addAction(saveAction)

        toolbar = self.addToolBar('Exit')
        toolbar.addAction(clearAction)
        toolbar.addAction(doodleAction)
        toolbar.addAction(writeAction)
        toolbar.addAction(classifyAction)
        toolbar.addAction(beautifyAction)
        toolbar.addAction(undoAction)

        self.combo = QComboBox()
        toolbar.addWidget(self.combo)
        self.styles = self.get_styles()
        self.combo.insertItems(1, self.styles)

        # list = QListWidget()
        # itm = QListWidgetItem("Tick")
        # # itm.setIcon(QIcon(r"tick.png"))
        # list.addItem(itm)
        # list.addItem(QListWidgetItem("fire"))
        # toolbar.addWidget(list)


        # self.setGeometry(300, 300, 350, 250)
        self.show()

    def get_available_objects(self):
        return os.listdir(PIC_DIR)

    def get_temp_file(self):
        return os.path.join(TEMP_DIR, f"temp_{self.id}.jpg")

    def get_styles(self):
        res = []
        for f in os.listdir("styles"):
            if f.split(".")[-1] == "ckpt":
                res.append(f.split(".")[0])
        return res

    def beautify(self):
        self.temp_save()
        out_path = os.path.join(OUTPUT_DIR, "out.jpg")
        style = self.combo.currentText()
        style_path = os.path.join("styles", style+".ckpt")
        ffwd_to_img(self.get_temp_file(), out_path, style_path)
        self.replace_image(out_path)

    def temp_save(self):
        self.id += 1
        self.image.save(self.get_temp_file())
        shutil.copyfile(STROKE_FILE, os.path.join(STROKE_DIR, f"out_{self.id}.csv"))

    def undo(self):
        if self.id == 0:
            return
        self.id -= 1
        print(self.get_temp_file())
        self.replace_image(self.get_temp_file())
        shutil.copyfile(os.path.join(STROKE_DIR, f"out_{self.id}.csv"), STROKE_FILE) 


    def start_doodling(self):
        with open(STROKE_FILE, "w") as f:
            f.write("")
        self.temp_save()
        self.start_doodle_id = self.id
        self.doodling = True

    def start_writing(self):
        print(self.combo.currentText())
        self.replace_image(self.get_temp_file())
        with open(STROKE_FILE, "w") as f:
            f.write("")
        self.doodling = False

    def classify(self):
        if os.stat(STROKE_FILE).st_size == 0:
            return
        result = combine_quickdraw(STROKE_FILE)
        top = self.get_best_result(result)
        print(result)
        if top is None:
            self.replace_image(os.path.join(TEMP_DIR, f"temp_{self.start_doodle_id}.jpg"))
            with open(STROKE_FILE, "w") as f:
                f.write("")
            return
        pic_path = get_pic_path_by_name(top)
        print('Pic path:', pic_path)
        x1, y1, x2, y2 = find_bounding_box()
        rect = QRect(x1, y1, x2-x1, y2-y1)
        print(x1, y1, x2, y2)
        
        self.replace_image(os.path.join(TEMP_DIR, f"temp_{self.start_doodle_id}.jpg"))
        self.insert_pic(pic_path, rect)
        self.temp_save()
        with open(STROKE_FILE, "w") as f:
            f.write("")

    def get_best_result(self, result):
        for c in result:
            if c in self.types:
                return c
        return


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
        if self.doodling:
            with open(STROKE_FILE, "a") as f:
                f.write("-1, -1\n")
            self.temp_save()
        else:
            self.selection_end = (e.x(), e.y())
            # painter = QtGui.QPainter(self.label.pixmap())
            # pen = QtGui.QPen()
            # pen.setWidth(10)
            # pen.setColor(QtGui.QColor('red'))
            # painter.setPen(pen)

            w = self.selection_end[0] - self.selection_start[0]
            h = self.selection_end[1] - self.selection_start[1]

            self.getText()
            pic_path = get_pic_path_by_name(self.obj_name)
            if pic_path is None:
                return
            rect = QRect(QPoint(*self.selection_start),QPoint(*self.selection_end))
            self.insert_pic(pic_path, rect)
            self.temp_save()
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
        if self.doodling:
            # painter = QtGui.QPainter(self.label.pixmap())
            painter = QtGui.QPainter(self.image)
            pen = QtGui.QPen()
            pen.setWidth(10)
            pen.setColor(QtGui.QColor('red'))
            painter.setPen(pen)
            painter.drawPoint(e.x(), e.y(), )
            if time.time() - self.time > 0.2:
                self.time = time.time()
                with open(STROKE_FILE, "a") as f:
                    f.write(f"{e.x()}, {e.y()}\n")
            painter.end()
            self.update()

    def paintEvent(self, e):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

    def replace_image(self, path):
        pic = QtGui.QPixmap(path)
        canvasPainter = QPainter(self.image)
        canvasPainter.drawPixmap(self.rect(), pic)
        self.update()
        # canvasPainter.drawImage(self.rect(), pic, pic.rect())
        # painter.drawPixmap(self.rect(), QPixmap("ninja.png"))

    def clear(self):
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())
        self.update()
        self.temp_save()

    def get_style(self):
        return
        # QInputDialog.comboBoxItems()
        # window = QDialog(self)
        # list = QListWidget(window)
        # listWidget = QListWidget(list)
        #
        # QListWidgetItem("Geeks", listWidget)
        # QListWidgetItem("For", listWidget)
        # QListWidgetItem("Geeks", listWidget)
        #
        # listWidgetItem = QListWidgetItem("GeeksForGeeks")
        # listWidget.addItem(listWidgetItem)
        # self.update()
        # self.window().add


def get_pic_by_name(name):
    name_dir = os.path.join(PIC_DIR, name)
    file = None
    for f in os.listdir(name_dir): 
        # print('File:', f)
        if f.split('.')[-1] in ["jpg", "png", "jpeg"]:
            file = f
            break
    img = cv2.imread(os.path.join(name_dir,file))
    return img

def get_pic_path_by_name(name):
    if name is None:
        return
    name_dir = os.path.join(PIC_DIR, name)
    if not os.path.isdir(name_dir):
        print(f"{name} is not yet supported")
        return
    file = None
    for f in os.listdir(name_dir): 
        # print('File:', f)
        if f.split('.')[-1] in ["jpg", "png", "jpeg"]:
            file = f
            break
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