import ctypes
import math

import numpy


def test():
    bs = [0x10, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xed, 0xff]
    s = 0
    i = 0

    while (i + 1) < 13:
        s += bs[i]
        s += bs[i + 1] << 8
        i += 2
    print(s)
    if (i + 1) == 13:
        s += bs[i]
    print(s)
    s = s & 0xffff + (s >> 16)
    print(s)

    print(~s & 0xffff)
    s = ~s & 0xffff
    print(s)


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
    return (~s & 0xffff)
    # s = ~s & 0xffff
    # return s


def test1():
    bs = [0x10, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xed, 0xff]
    val = calc_chksum(bs, 13)
    print(val)


def select_armor(i):
    radius = 10
    x = 451 + radius * math.cos(math.pi - i * 2 * math.pi / 4)
    y = 624 - radius * math.sin(math.pi - i * 2 * math.pi / 4)
    print(451, ",", 624, ",", round(x), ",", round(y))


def select_props(i):
    radius = 10
    x = 821 + radius * math.cos(1.5 * math.pi - (i + 1) * 2 * math.pi / 7)
    y = 627 - radius * math.sin(1.5 * math.pi - (i + 1) * 2 * math.pi / 7)
    print(821, ",", 627, ",", round(x), ",", round(y))


def select_mark(i):
    radius = 10
    x = 1110 + radius * math.cos(1.5 * math.pi + i * 2 * math.pi / 4)
    y = 228 - radius * math.sin(1.5 * math.pi + i * 2 * math.pi / 4)
    print(1110, ",", 228, ",", round(x), ",", round(y))


def test2():
    for i in range(6):
        select_props(i)
    for i in range(3):
        select_armor(i)
    for i in range(4):
        select_mark(i)


class BBB():
    b = 1

    def test(self):
        print('test', self.b, BBB.b)
        self.b += 1
        print('test', self.b, BBB.b)

    def test1(self):
        print('test1', self.b, BBB.b)
        BBB.b += 1
        print('test1', self.b, BBB.b)


class AAA():
    def __init__(self):
        if not hasattr(AAA, 'main'):
            AAA.main = BBB()


def test3():
    # test2()
    a = AAA()
    # print(AAA.main.b)
    AAA.main.test()


def test4():
    a = [False, False, True]
    b = [True, False, True]
    c = [True, False, True]
    d = [(1, 2), (3, 4)]
    e = [(1, 2), (3, 6)]
    print(a != b, b == c, d == e)

    f = e
    e = [(1, 2), (3, 4)]
    print(f == e)
    print(round(1.8))


def test5():
    b = BBB()
    b.test1()
    m = [1, 2]
    print(m, m[1])
    m = [False] * 4
    print(m)

    m = 0
    k, m, n = 1, 2, 3
    print(m)

    b = [[1, 2, 3], [4, 5, 6]]
    a = [j + 1 for i in b for j in i]

    print(a)

    e = [(1, 2), [3, 4]]
    f = (1, 2), [3, 4]
    print(e[0], e[1])
    print(e[0][0], e[1][1])
    print(f[0], f[1])
    print(f[0][0], f[1][1])


mc_width = 1280
mc_height = 720

wd_left = 34
wd_top = 38
wd_width = 1212
wd_high = 682


def cnvt_x(x):
    return x * wd_width / mc_width


def cnvt_y(y):
    return y * wd_high / mc_height


def cnvt_circles_0(circles):
    return [[cnvt_x(c[0]), cnvt_y(c[1]), cnvt_x(c[2])] for c in circles]


def cnvt_rect(left, top, right, bottom):
    return wd_left + cnvt_x(left), wd_top + cnvt_y(top), wd_left + cnvt_x(right), wd_top + cnvt_y(bottom)


moto = [[79, 190, 40], [79, 59, 41], [201, 55, 41]]
n = cnvt_circles_0([[79, 190, 40], [79, 59, 41], [201, 55, 41]])
print(moto, n)

left, top, right, bottom = [int(i) for i in cnvt_rect(681, 283, 933, 571)]
print(left, top, right, bottom)


def te(a):
    print('te', a)


a = (0, 1)


def ad(a):
    return a[0] + 1, a[1] + 1


te(a)
te(ad(a))
b = ad(a)
te(b)

a = [[0] * 2 for i in range(6)]
b = [[0, 0], [1, 0], [2, 0], [0, 0]]
print(a + b)

a = [False] * 3
b = a.copy()
a[0] = True
print(a, b)
bak_f_x, bak_f_y = 0, 0
print(bak_f_x, bak_f_y)

test2()
a = numpy.array([1, 2, 3])
b = (1, 1, 2)
print(a + 1)


class M:
    m = 123


print(M.m)

g = 1


def change_g():
    global g
    g = 3


print(g)
change_g()
print(g)
import ctypes


class Ct(ctypes.Structure):
    _fields_ = [('a', ctypes.c_int), ('b', ctypes.c_int)]


print(Ct.b.offset, Ct.b.size)
m = bytes([2, 0, 0, 0, 5, 0, 0, 0])

c = Ct()
d = Ct()
ctypes.memmove(ctypes.pointer(c), m, ctypes.sizeof(Ct))
ctypes.memmove(ctypes.byref(d), m, ctypes.sizeof(Ct))
print(c.a, c.b)
print(d.a, d.b)
v=ctypes.c_uint16(100)
print(ctypes.pointer(v) , ctypes.byref(v))
n = bytes([1,0,0,0])
print(int.from_bytes(n,'little'))

