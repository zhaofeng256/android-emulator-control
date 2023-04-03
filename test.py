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
    b = 1
    def test(self):
        print('test', self.b, BBB.b)
        self.b+=1
        print('test', self.b, BBB.b)
    def test1(self):
        print('test1', self.b, BBB.b)
        BBB.b+=1
        print('test1', self.b, BBB.b)
class AAA():
    def __init__(self):
        if not hasattr(AAA, 'main'):
            AAA.main = BBB()

def test3():
    #test2()
    a = AAA()
    #print(AAA.main.b)
    AAA.main.test()

def test4():
    a = [False, False, True]
    b= [True, False, True]
    c= [True, False, True]
    d = [(1,2),(3,4)]
    e = [(1,2),(3,6)]
    print(a != b, b ==c, d==e)

    f = e
    e = [(1,2),(3,4)]
    print(f==e)
    print(round(1.8))

b = BBB()
b.test1()
m = [1,2]
print(m, m[1])
m = [False]*4
print(m)

m = 0
k, m ,n = 1,2,3
print(m)