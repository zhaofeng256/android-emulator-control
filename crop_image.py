import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from screenshot import save_circle_panel, save_panel_axis

class Canvas(QWidget):

    def __init__(self, wnd, photo, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image = None
        self.shape = 0
        self.radius = 0
        self.center = QPoint()
        self.start = QPoint()
        self.end = QPoint()
        self.window = wnd

        self.pressed = self.moving = False
        # self.revisions = []

        # QShortcut(QKeySequence('Ctrl+Z'), self, self.undo)
        # QShortcut(QKeySequence('Ctrl+R'), self, self.reset)
        QShortcut(Qt.Key_Left, self, self.move_left)
        QShortcut(Qt.Key_Right, self, self.move_right)
        QShortcut(Qt.Key_Up, self, self.move_up)
        QShortcut(Qt.Key_Down, self, self.move_down)
        QShortcut(Qt.Key_Plus, self, self.zoom_in)
        QShortcut(Qt.Key_Minus, self, self.zoom_out)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.pressed = True
            self.radius = 0
            self.center = event.pos()
            self.start = event.pos()
            self.end = event.pos()
            self.update_position()
            self.update()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.moving = True
            r = (event.pos().x() - self.center.x()) ** 2 + (event.pos().y() - self.center.y()) ** 2
            self.radius = r ** 0.5
            if self.shape == 1:
                self.end = event.pos()
            elif self.shape == 2:
                side_len = event.pos().x() - self.start.x()
                self.end = QPoint(self.start.x() + side_len, self.start.y() + side_len)
            self.update_position()
            self.update()

    # def mouseReleaseEvent(self, event):
    #     if event.button() == Qt.LeftButton:
    #         self.pressed = self.moving = False
    #         self.revisions.append(self.image.copy())
    #         qp = QPainter(self.image)
    #         if self.moving:
    #             if self.shape == 0:
    #                 self.draw_circle(qp)
    #             elif self.shape == 1:
    #                 self.draw_rectangle(qp)
    #             elif self.shape == 2:
    #                 self.draw_square(qp)
    #         else:
    #             self.draw_point(qp)
    #
    #         self.update_position()
    #         self.update()

    # def keyPressEvent(self, event):
    #     self.keyPressed.emit(event)
    #     print('event.key()')
    #     event.accept()

    def paintEvent(self, event):
        qp = QPainter(self)
        rect = event.rect()
        qp.drawImage(rect, self.image, rect)
        # if self.moving:
        if self.shape == 0:
            self.draw_circle(qp)
        elif self.shape == 1:
            self.draw_rectangle(qp)
        elif self.shape == 2:
            self.draw_square(qp)
        # elif self.pressed:
        #     self.draw_point(qp)

    def draw_point(self, qp):
        qp.setPen(QPen(Qt.white, 2))
        qp.drawPoint(self.center)

    def draw_circle(self, qp):
        qp.setRenderHint(QPainter.Antialiasing)
        qp.setPen(QPen(Qt.white, 1, Qt.DashLine))
        qp.drawEllipse(self.center, self.radius, self.radius)

    def draw_rectangle(self, qp):
        qp.setRenderHint(QPainter.Antialiasing)
        qp.setPen(QPen(Qt.white, 1, Qt.DashLine))
        qp.drawRect(QRect(self.start, self.end))

    def draw_square(self, qp):
        self.draw_rectangle(qp)

    # def undo(self):
    #     if self.revisions:
    #         self.image = self.revisions.pop()
    #         self.update()
    #
    # def reset(self):
    #     if self.revisions:
    #         self.image = self.revisions[0]
    #         self.revisions.clear()
    #         self.update()

    def move(self):
        qp = QPainter(self)
        rect = self.image.rect()
        qp.drawImage(rect, self.image, rect)
        if self.shape == 0:
            self.draw_circle(qp)
        elif self.shape == 1:
            self.draw_rectangle(qp)
        elif self.shape == 2:
            self.draw_square(qp)
        self.update_position()
        self.update()

    def move_left(self):
        self.center = QPoint(self.center.x() - 1, self.center.y())
        self.start = QPoint(self.start.x() - 1, self.start.y())
        self.end = QPoint(self.end.x() - 1, self.end.y())
        self.move()

    def move_right(self):
        self.center = QPoint(self.center.x() + 1, self.center.y())
        self.start = QPoint(self.start.x() + 1, self.start.y())
        self.end = QPoint(self.end.x() + 1, self.end.y())
        self.move()

    def move_up(self):
        self.center = QPoint(self.center.x(), self.center.y() - 1)
        self.start = QPoint(self.start.x(), self.start.y() - 1)
        self.end = QPoint(self.end.x(), self.end.y() - 1)
        self.move()

    def move_down(self):
        self.center = QPoint(self.center.x(), self.center.y() + 1)
        self.start = QPoint(self.start.x(), self.start.y() + 1)
        self.end = QPoint(self.end.x(), self.end.y() + 1)
        self.move()

    def zoom_in(self):
        self.radius += 1
        self.end = QPoint(self.end.x() + 1, self.end.y() + 1)
        self.move()

    def zoom_out(self):
        self.radius -= 1
        self.end = QPoint(self.end.x() - 1, self.end.y() - 1)
        self.move()

    def update_position(self):
        if self.shape == 0:
            start_x = self.center.x() - self.radius
            start_y = self.center.y() - self.radius
            end_x = self.center.x() + self.radius
            end_y = self.center.y() + self.radius
            txt = "{start_x},{start_y},{end_x},{end_y}".format(start_x=round(start_x), start_y=round(start_y),
                                                               end_x=round(end_x), end_y=round(end_y))
            self.window.cut_position.setText(txt)
            print(txt)
        elif self.shape == 1 or self.shape == 2:
            start_x = self.start.x()
            start_y = self.start.y()
            end_x = self.end.x()
            end_y = self.end.y()
            txt = "{start_x},{start_y},{end_x},{end_y}".format(start_x=start_x, start_y=start_y,
                                                               end_x=end_x, end_y=end_y)
            self.window.cut_position.setText(txt)
            print(txt)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.canvas = None
        self.file = None
        self.out_folder = 'crop/'
        if not QFileInfo(self.out_folder).exists():
            os.mkdir(self.out_folder)

        w = QWidget()
        self.setCentralWidget(w)
        grid = QGridLayout(w)

        self.get_file = QPushButton('select')
        grid.addWidget(self.get_file)
        self.circle = QtWidgets.QRadioButton('circle')
        grid.addWidget(self.circle)
        self.circle.setChecked(True)
        self.rectangle = QtWidgets.QRadioButton('rectangle')
        grid.addWidget(self.rectangle)
        self.rectangle.setChecked(False)
        self.square = QtWidgets.QRadioButton('square')
        grid.addWidget(self.square)
        self.square.setChecked(False)
        self.key_code = QtWidgets.QLineEdit('key_code')
        self.key_code.setValidator(QIntValidator(0, 99))
        grid.addWidget(self.key_code)
        self.v_id = QtWidgets.QLineEdit('id')
        self.v_id.setValidator(QIntValidator(0, 99))
        grid.addWidget(self.v_id)
        self.detect_method = QtWidgets.QLineEdit('method')
        self.detect_method.setValidator(QIntValidator(0, 9))
        grid.addWidget(self.detect_method)
        self.cut = QtWidgets.QPushButton('cut')
        grid.addWidget(self.cut)
        self.cut_start = QPoint()
        self.cut_end = QPoint()
        self.cut_position = QtWidgets.QLabel('', self)
        grid.addWidget(self.cut_position)

        self.get_file.clicked.connect(self.on_clicked_select)
        self.circle.clicked.connect(self.on_clicked_circle)
        self.rectangle.clicked.connect(self.on_clicked_rectangle)
        self.square.clicked.connect(self.on_clicked_square)
        self.cut.clicked.connect(self.on_clicked_cut)

        self.canvas = Canvas(self, self.file)
        # grid.addWidget(self.canvas)
        self.canvas.setWindowFlag(Qt.FramelessWindowHint)

        self.on_clicked_select()

    def closeEvent(self, event):
        self.canvas.close()

    def on_clicked_select(self):
        options = QtWidgets.QFileDialog.Options()
        file, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "select file", "~",
                                                         "PNG Files (*.png)", options=options)
        self.file = file[0]
        print(self.file)
        self.canvas.image = QImage(self.file)
        self.canvas.setFixedSize(self.canvas.image.width(), self.canvas.image.height())
        self.canvas.update()
        self.canvas.show()

    def on_clicked_circle(self):
        state = self.circle.isChecked()
        if state:
            self.rectangle.setChecked(False)
            self.square.setChecked(False)
            self.canvas.shape = 0

    def on_clicked_rectangle(self):
        state = self.rectangle.isChecked()
        if state:
            self.circle.setChecked(False)
            self.square.setChecked(False)
            self.canvas.shape = 1

    def on_clicked_square(self):
        state = self.square.isChecked()
        if state:
            self.circle.setChecked(False)
            self.rectangle.setChecked(False)
            self.canvas.shape = 2

    def on_clicked_cut(self):
        base = os.path.basename(self.file)
        dst_file = self.out_folder + base
        prefix = os.path.splitext(base)[0]
        if self.canvas.shape == 0:
            save_circle_panel(self.file, self.canvas.center.x(), self.canvas.center.y(), self.canvas.radius, dst_file)
            start_x = self.canvas.center.x() - self.canvas.radius
            start_y = self.canvas.center.y() - self.canvas.radius
            end_x = self.canvas.center.x() + self.canvas.radius
            end_y = self.canvas.center.y() + self.canvas.radius
            save_panel_axis(int(self.key_code.text()), int(self.v_id.text()), prefix,
                        start_x, start_y, end_x, end_y, int(self.detect_method.text()))

        elif self.canvas.shape == 1 or self.canvas.shape == 2:
            image = cv2.imread(self.file)
            crop = image[self.canvas.start.y():self.canvas.end.y(), self.canvas.start.x():self.canvas.end.x()]
            cv2.imwrite(dst_file, crop)
            start_x = self.canvas.start.x()
            start_y = self.canvas.start.y()
            end_x = self.canvas.end.x()
            end_y = self.canvas.end.y()
            print(int(self.key_code.text()), int(self.v_id.text()), os.path.basename(self.file), prefix,
                        start_x, start_y, end_x, end_y, int(self.detect_method.text()))
            save_panel_axis(int(self.key_code.text()), int(self.v_id.text()), prefix,
                        start_x, start_y, end_x, end_y, int(self.detect_method.text()))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = MainWindow()
    gui.show()
    sys.exit(app.exec_())
