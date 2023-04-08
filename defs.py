from ctypes import Structure, memmove, byref, c_char, c_int32, sizeof, c_int16

TCP_PORT = 65432


class TcpData(Structure):
    _fields_ = [
        ("id", c_char * 4),
        ("type", c_char),
        ("param1", c_char * 4),
        ("param2", c_char * 4),
        ("checksum", c_char * 2),
    ]


def set_id(data, val):
    memmove(byref(data, TcpData.id.offset), byref(c_int32(val)), TcpData.id.size)


def set_param1_int32(data, val):
    memmove(byref(data, TcpData.param1.offset), byref(c_int32(val)), TcpData.param1.size)


def set_param2_int32(data, val):
    memmove(byref(data, TcpData.param2.offset), byref(c_int32(val)), TcpData.param2.size)


def get_param1_int32(data):
    b = data[TcpData.param1.offset:TcpData.param1.offset + TcpData.param1.size]
    return int.from_bytes(b, "little")


def get_param2_int32(data):
    b = data[TcpData.param2.offset:TcpData.param2.offset + TcpData.param2.size]
    return int.from_bytes(b, "little")


def set_checksum(data):
    sz = sizeof(data)
    if sz == sizeof(TcpData):
        val = calc_checksum(data, sz - 2)
        memmove(byref(data, TcpData.checksum.offset), byref(val), TcpData.checksum.size)
    else:
        print('set checksum len error')


def calc_checksum(data, length):
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
    return c_int16(~s & 0xffff)


class EventType:
    TYPE_KEYBOARD = 0
    TYPE_MOUSE_AXIS = 1
    TYPE_MOUSE_BUTTON = 2
    TYPE_MOUSE_WHEEL = 3
    TYPE_CONTROL = 4
    TYPE_SET_KEY_MOTION = 5
    TYPE_SET_WINDOW = 6


class KeyEvent:
    KEY_UP = 0
    KEY_DOWN = 1


class ButtonType:
    LEFT = 0
    RIGHT = 1
    MIDDLE = 2
    BACK = 3
    FORWARD = 4


class WheelEvent:
    ROLL_BACK = 0
    ROLL_FORWARD = 1


class ControlEvent:
    MAIN_MODE = 0
    SUB_MODE = 1
    MAP_MODE = 2
    TRANSPARENT_MODE = 3


class MainModeType:
    MULTI_PLAYER = 0
    BATTLE_GROUND = 1
    PVE = 2


class SubModeType:
    NONE_SUB_MODE = 0
    SUB_MODE_OFFSET = 1
    DRIVE_MOTO = 1
    DRIVE_CHOPPER = 2
    DRIVE_COYOTE = 3


class MapModeStatus:
    MAP_MODE_OFF = 0
    MAP_MODE_ON = 1


class TransPointStatus:
    TRANSPARENT_OFF = 0
    TRANSPARENT_ON = 1


class SupplyType:
    SUPPLY_NONE = 0
    SUPPLY_RANDOM = 1
    SUPPLY_SYSTEM = 2
    SUPPLY_CUSTOM = 3


class SettingType:
    WINDOW_POS = 0
    WINDOW_SIZE = 1


class MotionType:
    MOTION_NONE = 0
    MOTION_TAP = 1
    MOTION_SYNC = 2
    MOTION_DRAG = 3
    MOTION_COMB = 4
    MOTION_TRANS = 5
