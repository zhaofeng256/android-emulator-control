import sys
import cv2
from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class Canvas(QWidget):

    def __init__(self, wnd, photo, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = 0
        self.radius = 0
        self.center = QPoint()
        self.start = QPoint()
        self.end = QPoint()
        self.window = wnd
        self.image = QImage(photo)
        self.setFixedSize(self.image.width(), self.image.height())
        self.pressed = self.moving = False
        self.revisions = []

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.pressed = True
            self.center = event.pos()
            self.start = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.moving = True
            r = (event.pos().x() - self.center.x()) ** 2 + (event.pos().y() - self.center.y()) ** 2
            self.radius = r ** 0.5
            self.end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.revisions.append(self.image.copy())
            qp = QPainter(self.image)
            if self.moving:
                if self.shape == 0:
                    self.draw_circle(qp)
                elif self.shape == 1:
                    self.draw_rectangle(qp)
            else:
                self.draw_point(qp)
            self.pressed = self.moving = False
            self.update_position()
            self.update()

    # def keyPressEvent(self, event):
    #     self.keyPressed.emit(event)
    #     print('event.key()')
    #     event.accept()

    def paintEvent(self, event):
        qp = QPainter(self)
        rect = event.rect()
        qp.drawImage(rect, self.image, rect)
        if self.moving:
            if self.shape == 0:
                self.draw_circle(qp)
            elif self.shape == 1:
                self.draw_rectangle(qp)
        elif self.pressed:
            self.draw_point(qp)

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

    def undo(self):
        if self.revisions:
            self.image = self.revisions.pop()
            self.update()

    def reset(self):
        if self.revisions:
            self.image = self.revisions[0]
            self.revisions.clear()
            self.update()

    def move(self):
        if self.revisions:
            self.undo()
            self.revisions.append(self.image.copy())
            print(self.center)
            qp = QPainter(self.image)
            if self.shape == 0:
                self.draw_circle(qp)
            elif self.shape == 1:
                self.draw_rectangle(qp)
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

    def update_position(self):
        if self.shape == 0:
            start_x = self.center.x() - self.radius
            start_y = self.center.y() - self.radius
            end_x = self.center.x() + self.radius
            end_y = self.center.y() + self.radius
            txt = "{start_x},{start_y},{end_x},{end_y}".format(start_x=start_x, start_y=start_y,
                                                                             end_x=end_x, end_y=end_y)
            self.window.cut_position.setText(txt)
            print(txt)
        elif self.shape == 1:
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
        w = QWidget()
        self.setCentralWidget(w)
        self.file = 'moto.png'
        self.out_folder = '4/'
        self.canvas = Canvas(self, self.file)
        grid = QGridLayout(w)
        grid.addWidget(self.canvas)
        self.circle = QtWidgets.QCheckBox('circle')
        grid.addWidget(self.circle)
        self.circle.setChecked(True)
        self.rectangle = QtWidgets.QCheckBox('rectangle')
        grid.addWidget(self.rectangle)
        self.rectangle.setChecked(False)
        self.cut = QtWidgets.QPushButton('cut')
        grid.addWidget(self.cut)
        self.cut_start = QPoint()
        self.cut_end = QPoint()
        self.cut_position = QtWidgets.QLabel('', self)
        grid.addWidget(self.cut_position)
        self.cut_position.setText('456')
        QShortcut(QKeySequence('Ctrl+Z'), self, self.canvas.undo)
        QShortcut(QKeySequence('Ctrl+R'), self, self.canvas.reset)
        QShortcut(Qt.Key_Left, self, self.canvas.move_left)
        QShortcut(Qt.Key_Right, self, self.canvas.move_right)
        QShortcut(Qt.Key_Up, self, self.canvas.move_up)
        QShortcut(Qt.Key_Down, self, self.canvas.move_down)
        self.circle.toggled.connect(self.on_clicked_circle)
        self.rectangle.toggled.connect(self.on_clicked_rectangle)
        self.cut.clicked.connect(self.on_clicked_cut)

    def on_clicked_circle(self):
        state = self.circle.isChecked()
        self.rectangle.setChecked(not state)
        if state:
            self.canvas.shape = 0
        else:
            self.canvas.shape = 1

    def on_clicked_rectangle(self):
        state = self.rectangle.isChecked()
        self.circle.setChecked(not state)
        if state:
            self.canvas.shape = 1
        else:
            self.canvas.shape = 0

    def on_clicked_cut(self):
        image = cv2.imread(self.file)
        crop = image[self.canvas.start.y():self.canvas.end.y(), self.canvas.start.x():self.canvas.end.x()]
        cv2.imwrite(self.out_folder + self.file, crop)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = MainWindow()
    gui.show()
    sys.exit(app.exec_())
