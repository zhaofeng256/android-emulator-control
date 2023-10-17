import ctypes
import re
import subprocess
import sys
import threading
import time

import mouse
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, QPoint, QMargins
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton, QComboBox, QStatusBar, \
    QMainWindow

import defs
# from dectect_mode import DetectModeService
# from keyborad_service import KeyboardService
# from mouse_service import MouseService
# from tcp_service import TcpServerService
# from window_info import get_window_info, update_window_info


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__(parent=None)
        self.statusBar = QStatusBar()
        if not hasattr(MainWindow, 'sss'):
            MainWindow.sss = self.statusBar
        self.setStatusBar(self.statusBar)
        self.setWindowTitle("android emulator control")
        self.move(
            QApplication.desktop().screen().rect().bottomRight() - self.rect().center() - QPoint(30, 100)
        )

    def closeEvent(self, event):
        mouse.unhook_all()
        if self.centralWidget().transparent_window:
            self.centralWidget().transparent_window.close()
        event.accept()


class WinForm(QWidget):
    def __init__(self):
        super(WinForm, self).__init__(parent=None)

        layout = QGridLayout()
        layout.setColumnStretch(3, 5)
        self.setLayout(layout)

        self.transparent_window = 0
        self.show_transparent_window = True

        self.cell_width = 100
        self.cell_high = 40

        self.combobox_devices = QComboBox()
        self.combobox_devices.setFixedSize(2 * self.cell_width + 5, self.cell_high)
        # combo text align center
        self.combobox_devices.setEditable(True)
        self.combobox_devices.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        self.combobox_devices.lineEdit().setReadOnly(True)
        layout.addWidget(self.combobox_devices, 0, 0, 1, 2, QtCore.Qt.AlignLeft)

        self.bt_refresh = QPushButton("refresh", self)
        self.bt_refresh.setFixedSize(self.cell_width, self.cell_high)
        self.bt_refresh.clicked.connect(self.bt_refresh_clicked)
        layout.addWidget(self.bt_refresh, 0, 2, 1, 1, QtCore.Qt.AlignCenter)

        self.bt_run = QPushButton("run", self)
        self.bt_run.setFixedSize(self.cell_width, self.cell_high)
        self.bt_run.clicked.connect(self.bt_run_clicked)
        layout.addWidget(self.bt_run, 1, 0, 1, 1, QtCore.Qt.AlignCenter)

        self.bt_stop_run = QPushButton("stop_run", self)
        self.bt_stop_run.setFixedSize(self.cell_width, self.cell_high)
        self.bt_stop_run.clicked.connect(self.bt_stop_run_clicked)
        layout.addWidget(self.bt_stop_run, 1, 1, 1, 1, QtCore.Qt.AlignCenter)

        self.bt_update_run = QPushButton("push_file", self)
        self.bt_update_run.setFixedSize(self.cell_width, self.cell_high)
        self.bt_update_run.clicked.connect(self.bt_update_run_clicked)
        layout.addWidget(self.bt_update_run, 1, 2, 1, 1, QtCore.Qt.AlignCenter)
        self.bt_update_run.setDisabled(True)

        self.bt_cross_hair = QPushButton("cross_hair", self)
        self.bt_cross_hair.setFixedSize(self.cell_width, self.cell_high)
        self.bt_cross_hair.clicked.connect(self.bt_cross_hair_clicked)
        layout.addWidget(self.bt_cross_hair, 2, 0, 1, 1, QtCore.Qt.AlignCenter)

        self.bt_point_position = QPushButton("point_position", self)
        self.bt_point_position.setFixedSize(self.cell_width, self.cell_high)
        self.bt_point_position.clicked.connect(self.bt_point_position_clicked)
        layout.addWidget(self.bt_point_position, 2, 1, 1, 1, QtCore.Qt.AlignCenter)

        self.bt_not_point = QPushButton("not_point", self)
        self.bt_not_point.setFixedSize(self.cell_width, self.cell_high)
        self.bt_not_point.clicked.connect(self.bt_not_point_clicked)
        layout.addWidget(self.bt_not_point, 2, 2, 1, 1, QtCore.Qt.AlignCenter)

        self.bt_stop_move = QPushButton("stop_move", self)
        self.bt_stop_move.setFixedSize(self.cell_width, self.cell_high)
        self.bt_stop_move.clicked.connect(self.bt_stop_move_clicked)
        layout.addWidget(self.bt_stop_move, 3, 0, 1, 1, QtCore.Qt.AlignCenter)

        self.bt_start_coyote = QPushButton("coyote", self)
        self.bt_start_coyote.setFixedSize(self.cell_width, self.cell_high)
        self.bt_start_coyote.clicked.connect(self.bt_start_coyote_clicked)
        layout.addWidget(self.bt_start_coyote, 3, 1, 1, 1, QtCore.Qt.AlignCenter)

        self.bt_stop_send = QPushButton("stop_send", self)
        self.bt_stop_send.setFixedSize(self.cell_width, self.cell_high)
        self.bt_stop_send.clicked.connect(self.bt_stop_send_clicked)
        layout.addWidget(self.bt_stop_send, 3, 2, 1, 1, QtCore.Qt.AlignCenter)

        self.bt_transparent_board = QPushButton("transparent", self)
        self.bt_transparent_board.setFixedSize(self.cell_width, self.cell_high)
        self.bt_transparent_board.clicked.connect(self.bt_transparent_board_clicked)
        layout.addWidget(self.bt_transparent_board, 4, 0, 1, 1, QtCore.Qt.AlignCenter)

        self.bt_home = QPushButton("home", self)
        self.bt_home.setFixedSize(self.cell_width, self.cell_high)
        self.bt_home.clicked.connect(self.bt_home_clicked)
        layout.addWidget(self.bt_home, 4, 1, 1, 1, QtCore.Qt.AlignCenter)

        self.bt_jump = QPushButton("jump", self)
        self.bt_jump.setFixedSize(self.cell_width, self.cell_high)
        self.bt_jump.clicked.connect(self.bt_jump_clicked)
        layout.addWidget(self.bt_jump, 4, 2, 1, 1, QtCore.Qt.AlignCenter)

        self.bt_stop_jump = QPushButton("stop_jump", self)
        self.bt_stop_jump.setFixedSize(self.cell_width, self.cell_high)
        self.bt_stop_jump.clicked.connect(self.bt_stop_jump_clicked)
        layout.addWidget(self.bt_stop_jump, 5, 0, 1, 1, QtCore.Qt.AlignCenter)

        self.bt_walk = QPushButton("walk", self)
        self.bt_walk.setFixedSize(self.cell_width, self.cell_high)
        self.bt_walk.clicked.connect(self.bt_walk_clicked)
        layout.addWidget(self.bt_walk, 5, 1, 1, 1, QtCore.Qt.AlignCenter)

        self.bt_stop_walk = QPushButton("stop_walk", self)
        self.bt_stop_walk.setFixedSize(self.cell_width, self.cell_high)
        self.bt_stop_walk.clicked.connect(self.bt_stop_walk_clicked)
        layout.addWidget(self.bt_stop_walk, 5, 2, 1, 1, QtCore.Qt.AlignCenter)

        self.running = False
        self.bt_stop_run.setDisabled(True)

        self.jumping = False
        self.bt_stop_jump.setDisabled(True)

        self.walking = False
        self.bt_stop_walk.setDisabled(True)

    def bt_refresh_clicked(self):
        pipe = subprocess.Popen('adb kill-server', stdout=subprocess.PIPE)
        out = pipe.communicate()[0]
        s = out.decode()
        print('adb kill-server' + ':\n' + s)
        pipe.terminate()

        pipe = subprocess.Popen('adb devices', stdout=subprocess.PIPE)
        out = pipe.communicate()[0]
        s = out.decode()
        print('adb devices' + ':\n' + out.decode())
        pipe.terminate()

        pipe = subprocess.Popen('adb reconnect offline', stdout=subprocess.PIPE)
        out = pipe.communicate()[0]
        s = out.decode()
        print('adb reconnect offline' + ':\n' + s)
        pipe.terminate()

        pipe = subprocess.Popen("adb devices", stdout=subprocess.PIPE)
        out = pipe.communicate()[0]
        s = out.decode()
        x = re.findall('\r\n(.+?)\t', s)
        print('adb devices' + ':\n' + s)
        pipe.terminate()

        for i in range(self.combobox_devices.count()):
            self.combobox_devices.removeItem(0)

        for i in range(len(x)):
            print(x[i])
            self.combobox_devices.addItem(x[i])

    def bt_open_cross_hair(self):
        script = [
            'shell am start - n com.customscopecommunity.crosshairpro/.MainActivity',
            'shell sleep 2',
            'shell input tap 543 1132',
            'shell am start - n com.tencent.tmgp.cod/com.tencent.tmgp.cod.CODMainActivity'
        ]
        self.run_script(script)

    def thread_run(self):
        self.bt_run.setDisabled(True)
        self.bt_stop_run.setEnabled(True)

        if (not self.running):
            self.running = True
            while (self.running):
                self.run_script(['shell input tap 985 222'])
                time.sleep(5)
                self.run_script(['shell input tap 985 222'])
                self.run_script(['shell input touchscreen swipe 743 400 1150 400 500'])
                print('running...')

        self.bt_run.setEnabled(True)
        self.bt_stop_run.setDisabled(True)

    def bt_run_clicked(self):
        threading.Thread(target=self.thread_run).start()

    def bt_stop_run_clicked(self):
        self.running = False
        self.bt_stop_run.setDisabled(True)

    def bt_update_run_clicked(self):
        script = [
            r"push ./run.sh /data/local/tmp",
            "shell chmod a+x /data/local/tmp/run.sh",
            "shell sync"
        ]
        self.run_script(script)

    def bt_cross_hair_clicked(self):
        script = [
            "shell am start -n com.customscopecommunity.crosshairpro/.MainActivity",
            "shell sleep 2",
            "shell input tap 543 1132",
            "shell am start -n com.tencent.tmgp.cod/com.tencent.tmgp.cod.CODMainActivity"
        ]
        self.run_script(script)

    def bt_point_position_clicked(self):
        script = [
            "shell settings put global development_settings_enabled 1",
            "shell settings put system pointer_location 1",
            "shell settings put system show_touches 1"
        ]
        self.run_script(script)

    def bt_not_point_clicked(self):
        script = [
            "shell settings put global development_settings_enabled 1",
            "shell settings put system pointer_location 0",
            "shell settings put system show_touches  0"
        ]
        self.run_script(script)

    def bt_start_coyote_clicked(self):
        script = [
            'shell am start -n com.zf.coyote/com.zf.coyote.MainActivity',
            'shell sleep 2',
            'shell am start -n com.tencent.tmgp.cod/com.tencent.tmgp.cod.CODMainActivity'
        ]
        self.run_script(script)

    def bt_home_clicked(self):
        script = [
            'shell input keyevent 3'
        ]
        self.run_script(script)

    def run_script(self, script):
        device_name = self.combobox_devices.currentText()
        cmd = "adb -s " + device_name + " "

        for i in range(len(script)):
            print(cmd + script[i])
            pipe = subprocess.Popen(cmd + script[i], stdout=subprocess.PIPE)
            out = pipe.communicate()[0]
            print(out.decode())

    def bt_stop_move_clicked(self):
        if MouseService.stop_move == 0:
            MouseService.stop_move = 1
        else:
            MouseService.stop_move = 0

    def bt_stop_send_clicked(self):
        TcpServerService.stop_send = 1 - TcpServerService.stop_send

    def bt_transparent_board_clicked(self):
        self.show_transparent_window = not self.show_transparent_window

        x, y, w, h = get_window_info()
        self.transparent_window.move(x - 5, y - 5)
        self.transparent_window.setFixedSize(w + 10, h + 10)

        if self.show_transparent_window:
            self.transparent_window.show()
        else:
            self.transparent_window.hide()

    def thread_jump(self):
        self.bt_jump.setDisabled(True)
        self.bt_stop_jump.setEnabled(True)

        if not self.jumping:
            self.jumping = True
            self.bt_refresh.setDisabled(True)
            while self.jumping:
                script = (['shell input tap 1212 508'])

                cnt = self.combobox_devices.count()
                for n in range(cnt):
                    if n == 0:
                        continue

                    device_name = self.combobox_devices.itemText(n)
                    cmd = "adb -s " + device_name + " "

                    for i in range(len(script)):
                        print(cmd + script[i])
                        pipe = subprocess.Popen(cmd + script[i], stdout=subprocess.PIPE)
                        out = pipe.communicate()[0]
                        print(out.decode())

                time.sleep(1)
                print('jumping...')

        self.bt_jump.setEnabled(True)
        self.bt_stop_jump.setDisabled(True)
        self.bt_refresh.setEnabled(True)

    def bt_jump_clicked(self):
        threading.Thread(target=self.thread_jump).start()

    def bt_stop_jump_clicked(self):
        self.jumping = False
        self.bt_stop_jump.setDisabled(True)

    def thread_walk(self):
        self.bt_walk.setDisabled(True)
        self.bt_stop_walk.setEnabled(True)

        if not self.walking:
            self.walking = True
            self.bt_refresh.setDisabled(True)
            while self.walking:
                script = (['shell input touchscreen swipe 200 500 200 480 100',
                           'shell input touchscreen swipe 200 500 200 520 100',
                           'shell input tap 1212 508'])
                time.sleep(5)
                cnt = self.combobox_devices.count()
                for n in range(cnt):
                    if n == 0:
                        continue

                    device_name = self.combobox_devices.itemText(n)
                    cmd = "adb -s " + device_name + " "

                    for i in range(len(script)):
                        print(cmd + script[i])
                        pipe = subprocess.Popen(cmd + script[i], stdout=subprocess.PIPE)
                        out = pipe.communicate()[0]
                        print(out.decode())

                time.sleep(1)
                print('walking...')

        self.bt_walk.setEnabled(True)
        self.bt_stop_walk.setDisabled(True)
        self.bt_refresh.setEnabled(True)

    def bt_walk_clicked(self):
        threading.Thread(target=self.thread_walk).start()

    def bt_stop_walk_clicked(self):
        self.walking = False
        self.bt_stop_walk.setDisabled(True)

class TransparentWindow(QWidget):
    def paintEvent(self, event=None):
        painter = QPainter(self)
        painter.setOpacity(0)
        painter.setBrush(Qt.white)
        pen = QPen(Qt.red)
        pen.setWidth(5)
        painter.setPen(pen)
        rect = self.rect() - QMargins(5, 5, 5, 5)
        painter.drawRect(rect)
        painter.setOpacity(1)
        painter.drawText(self.rect(), Qt.AlignCenter, "Qt")


def find_main_window():
    # Global function to find the (open) QMainWindow in application
    app = QApplication.instance()
    for widget in app.topLevelWidgets():
        if isinstance(widget, QMainWindow):
            return widget
    return None


def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def main():
    if is_admin():

        app = QtWidgets.QApplication(sys.argv)
        wf = WinForm()
        wf.bt_refresh_clicked()

        # wf.transparent_window = TransparentWindow()
        # wf.transparent_window.setWindowFlags(Qt.FramelessWindowHint)
        # wf.transparent_window.setAttribute(Qt.WA_NoSystemBackground, True)
        # wf.transparent_window.setAttribute(Qt.WA_TranslucentBackground, True)

        # update_window_info()
        # x, y, w, h = get_window_info()
        # wf.transparent_window.move(x - 5, y - 5)
        # wf.transparent_window.setFixedSize(w + 10, h + 10)
        # # wf.transparent_window.show()

        mw = MainWindow()
        mw.setCentralWidget(wf)

        # TcpServerService('', defs.TCP_PORT).start()
        # MouseService.set_statusbar(mw.sss)  # wf.setStatusTip("pos:" + str(info.window_pos))
        # MouseService.start()
        # KeyboardService.start()
        # DetectModeService().start()

        mw.show()

        sys.exit(app.exec_())

    else:
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)


if __name__ == "__main__":
    main()
