import tcp_service
from defs import TcpData, EventType, set_param1_int32, set_param2_int32, ControlEvent, MapModeStatus, TransPointStatus, \
    MainModeType, SubModeType


class ModeInfo:
    main_mode = MainModeType.MULTI_PLAYER
    sub_mode = SubModeType.NONE_SUB_MODE
    map_mode_on = False
    transparent_mode_on = False


def main_mode_switch(mode):
    ModeInfo.main_mode = mode
    print("switch to main mode", mode)
    data = TcpData()
    data.type = EventType.TYPE_CONTROL
    set_param1_int32(data, ControlEvent.MAIN_MODE)
    set_param2_int32(data, int(mode))
    tcp_service.tcp_data_append(data)


def map_mode_switch():
    ModeInfo.map_mode_on = not ModeInfo.map_mode_on
    print("map mode on is", ModeInfo.map_mode_on)
    data = TcpData()
    data.type = EventType.TYPE_CONTROL
    set_param1_int32(data, ControlEvent.MAP_MODE)
    if ModeInfo.map_mode_on:
        set_param2_int32(data, MapModeStatus.MAP_MODE_ON)
    else:
        set_param2_int32(data, MapModeStatus.MAP_MODE_OFF)
    tcp_service.tcp_data_append(data)


def trans_point_mode_switch():
    ModeInfo.transparent_mode_on = not ModeInfo.transparent_mode_on
    print('transparent point mode on is', ModeInfo.transparent_mode_on)
    data = TcpData()
    data.type = EventType.TYPE_CONTROL
    set_param1_int32(data, ControlEvent.TRANSPARENT_MODE)
    if ModeInfo.transparent_mode_on:
        set_param2_int32(data, TransPointStatus.TRANSPARENT_ON)
    else:
        set_param2_int32(data, TransPointStatus.TRANSPARENT_OFF)
    tcp_service.tcp_data_append(data)
