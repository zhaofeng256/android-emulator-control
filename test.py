import math

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


    print( ~s & 0xffff)
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
    x = 451 + radius * math.cos(math.pi - i * 2*math.pi / 4)
    y = 624 - radius * math.sin(math.pi - i * 2*math.pi / 4)
    print(451, ",", 624,  ",",round(x), ",", round(y))



def select_props(i):
    radius = 10
    x = 821 + radius * math.cos(1.5*math.pi - (i+1) * 2*math.pi / 7)
    y = 627 - radius * math.sin(1.5*math.pi - (i+1) * 2*math.pi / 7)
    print(821, ",", 627,  ",",round(x), ",", round(y))

def test2():
    for i in range(6):
        select_props(i)
    for i in range(3):
        select_armor(i)
class BBB():
    b = 0
    def test(self):
        print('test')

class AAA():
    def __init__(self):
        if not hasattr(AAA, 'main'):
            AAA.main = BBB()


#test2()
a = AAA()
#print(AAA.main.b)
AAA.main.test()
