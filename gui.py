import ctypes
import subprocess
import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, QPoint, QSize
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLabel, QPushButton, QComboBox, QHBoxLayout
import subprocess
import re
import threading
import time

import defs
from keyborad_service import KeyboardService
from mouse_service import MouseService
from tcp_service import TcpServerService
from window_info import window_info_init, is_admin, WindowInfo, info


class WinForm(QWidget):
    def __init__(self):
        super(WinForm, self).__init__(parent=None)

        self.setWindowTitle("android emuulator control")
        self.move(
            QApplication.desktop().screen().rect().center() - self.rect().center()
        )

        layout = QGridLayout()
        layout.setColumnStretch(3, 4)
        self.setLayout(layout)

        self.cell_width = 100
        self.cell_hight = 40

        self.combobox_devices = QComboBox()
        self.combobox_devices.setFixedSize(2 * self.cell_width + 5, self.cell_hight)
        # combo text align center
        self.combobox_devices.setEditable(True)
        self.combobox_devices.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        self.combobox_devices.lineEdit().setReadOnly(True)
        layout.addWidget(self.combobox_devices, 0, 0, 1, 2, QtCore.Qt.AlignLeft)

        self.bt_refresh = QPushButton("refresh", self)
        self.bt_refresh.setFixedSize(self.cell_width, self.cell_hight)
        self.bt_refresh.clicked.connect(self.bt_refresh_clicked)
        layout.addWidget(self.bt_refresh, 0, 2, 1, 1, QtCore.Qt.AlignCenter)

        self.bt_run = QPushButton("run", self)
        self.bt_run.setFixedSize(self.cell_width, self.cell_hight)
        self.bt_run.clicked.connect(self.bt_run_clicked)
        layout.addWidget(self.bt_run, 1, 0, 1, 1, QtCore.Qt.AlignCenter)

        self.bt_stop_run = QPushButton("stop_run", self)
        self.bt_stop_run.setFixedSize(self.cell_width, self.cell_hight)
        self.bt_stop_run.clicked.connect(self.bt_stop_run_clicked)
        layout.addWidget(self.bt_stop_run, 1, 1, 1, 1, QtCore.Qt.AlignCenter)

        self.bt_update_run = QPushButton("push_file", self)
        self.bt_update_run.setFixedSize(self.cell_width, self.cell_hight)
        self.bt_update_run.clicked.connect(self.bt_update_run_clicked)
        layout.addWidget(self.bt_update_run, 1, 2, 1, 1, QtCore.Qt.AlignCenter)
        self.bt_update_run.setDisabled(True)

        self.bt_cross_hair = QPushButton("cross_hair", self)
        self.bt_cross_hair.setFixedSize(self.cell_width, self.cell_hight)
        self.bt_cross_hair.clicked.connect(self.bt_cross_hair_clicked)
        layout.addWidget(self.bt_cross_hair, 2, 0, 1, 1, QtCore.Qt.AlignCenter)

        self.bt_point_position = QPushButton("point_position", self)
        self.bt_point_position.setFixedSize(self.cell_width, self.cell_hight)
        self.bt_point_position.clicked.connect(self.bt_point_position_clicked)
        layout.addWidget(self.bt_point_position, 2, 1, 1, 1, QtCore.Qt.AlignCenter)

        self.bt_not_point = QPushButton("not_point", self)
        self.bt_not_point.setFixedSize(self.cell_width, self.cell_hight)
        self.bt_not_point.clicked.connect(self.bt_not_point_clicked)
        layout.addWidget(self.bt_not_point, 2, 2, 1, 1, QtCore.Qt.AlignCenter)

        self.bt_stop_move = QPushButton("stop_move", self)
        self.bt_stop_move.setFixedSize(self.cell_width, self.cell_hight)
        self.bt_stop_move.clicked.connect(self.bt_stop_move_clicked)
        layout.addWidget(self.bt_stop_move, 3, 0, 1, 1, QtCore.Qt.AlignCenter)


        self.bt_start_coyote = QPushButton("coyote", self)
        self.bt_start_coyote.setFixedSize(self.cell_width, self.cell_hight)
        self.bt_start_coyote.clicked.connect(self.bt_start_coyote_clicked)
        layout.addWidget(self.bt_start_coyote, 3, 1, 1, 1, QtCore.Qt.AlignCenter)

        self.bt_stop_send = QPushButton("stop_send", self)
        self.bt_stop_send.setFixedSize(self.cell_width, self.cell_hight)
        self.bt_stop_send.clicked.connect(self.bt_stop_send_clicked)
        layout.addWidget(self.bt_stop_send, 3, 2, 1, 1, QtCore.Qt.AlignCenter)

        self.running = False
        self.bt_stop_run.setDisabled(True)

    def bt_refresh_clicked(self):
        pipe = subprocess.Popen("adb devices", stdout=subprocess.PIPE)
        out = pipe.communicate()[0]
        s = out.decode()
        x = re.findall('\r\n(.+?)\t', s)

        for i in range(self.combobox_devices.count()):
            self.combobox_devices.removeItem(i)

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
            "shell input tap 543 113w2",
            "shell am start -n com.tencent.tmgp.cod/com.tencent.tmgp.cod.CODMainActivity"
        ]
        self.run_script(script)

    def bt_point_position_clicked(self):
        script = [
            "shell settings put global development_settings_enabled 1",
            "shell settings put system pointer_location 1"
        ]
        self.run_script(script)

    def bt_not_point_clicked(self):
        script = [
            "shell settings put global development_settings_enabled 1",
            "shell settings put system pointer_location 0"
        ]
        self.run_script(script)

    def bt_start_coyote_clicked(self):
        script = [
            'shell am start -n com.zf.coyote/com.zf.coyote.MainActivity',
            'shell sleep 2',
            'shell am start -n com.tencent.tmgp.cod/com.tencent.tmgp.cod.CODMainActivity'
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
        if (MouseService.stop_move == 0):
            MouseService.stop_move = 1
        else:
            MouseService.stop_move = 0

    def bt_stop_send_clicked(self):
        TcpServerService.stop_send = 1 - TcpServerService.stop_send


class TransparentWindow(QWidget):
    def paintEvent(self, event=None):
        painter = QPainter(self)
        painter.setOpacity(0.5)
        painter.setBrush(Qt.white)
        painter.setPen(QPen(Qt.white))
        painter.drawRect(self.rect())


def main():

    if is_admin():
        window_info_init()
        TcpServerService('', defs.TCP_PORT).start()
        MouseService.start()
        KeyboardService.start()

        app = QtWidgets.QApplication(sys.argv)
        wf = WinForm()
        wf.bt_refresh_clicked()
        wf.show()

        qw = TransparentWindow()
        qw.setWindowFlags(Qt.FramelessWindowHint)
        qw.setAttribute(Qt.WA_NoSystemBackground, True)
        qw.setAttribute(Qt.WA_TranslucentBackground, True)
        qw.move(info.window_pos[0], info.window_pos[1])
        qw.setFixedSize(info.window_size[0], info.window_size[1])
        qw.show()
        print(info.window_pos[0], info.window_size[0])

        sys.exit(app.exec_())

    else:
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)


if __name__ == "__main__":
    main()
