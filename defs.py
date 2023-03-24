from ctypes import Structure, memmove, byref, c_char, c_int32, sizeof

TCP_PORT = 65432


class TcpData(Structure):
    _fields_ = [
        ("id", c_char * 4),
        ("type", c_char),
        ("param1", c_char * 4),
        ("param2", c_char * 4),
        ("checksum", c_char * 2),
    ]


OFFSET_ID = 0
OFFSET_PARAM_1 = 5
OFFSET_PARAM_2 = 9
OFFSET_CHKSUM = 13


def set_id(data, val):
    memmove(byref(data, OFFSET_ID), byref((c_int32)(val)), 4)


def set_param1(data, val):
    memmove(byref(data, OFFSET_PARAM_1), byref((c_int32)(val)), 4)


def set_param2(data, val):
    memmove(byref(data, OFFSET_PARAM_2), byref((c_int32)(val)), 4)


def set_chksum(data):
    sz = sizeof(data)
    if sz == sizeof(TcpData):
        val = calc_chksum(data, sz - 2)
        memmove(byref(data, OFFSET_CHKSUM), byref((c_int32)(val)), 2)
    else:
        print('set checksum len error')

def calc_chksum(data, length):
    s = 0
    i = 0
    bs = bytearray(data)
    while (i + 1) < length:
        s += bs[i]
        s += bs[i + 1] << 8
        i += 2

    if (i + 1) == length:
        s += bs[i]

    s = s & 0xffff + (s >> 16)
    return ~s & 0xffff


class EventType():
    TYPE_KEYBOARD = 0x0
    TYPE_MOUSE = 0x1
    TYPE_BUTTON = 0x2
    TYPE_WHEEL = 0x3


class KeyEvent():
    KEY_UP = 0
    KEY_DOWN = 1


class ButtonType():
    LEFT = 0
    RIGHT = 1
    MIDDLE = 2
    BACK = 3
    FORWARD = 4


class WheelEvent():
    ROLL_BACK = 0
    ROLL_FORWARD = 1
