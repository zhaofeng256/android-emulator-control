
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


bs = [0x10, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xed, 0xff]
val = calc_chksum(bs, 13)
print(val)