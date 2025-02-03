from struct import unpack, pack, calcsize
import os
from random import randint
from itertools import chain
import numpy as np # pip install numpy
from time import time

# https://formats.kaitai.io/bmp/python.html
# https://entropymine.com/jason/bmpsuite/bmpsuite/html/bmpsuite.html

def f_unpack(file, format):
    L = calcsize(format)
    return unpack(format, file.read(L))

# calcsize("<IHHIIiiHHIIIIII") -> 52 (без magic!)

HeaderTypes = {
    12: "bitmap_core_header",
    40: "bitmap_info_header",
    52: "bitmap_v2_info_header",
    56: "bitmap_v3_info_header",
    64: "os2_2x_bitmap_header",
    108: "bitmap_v4_header",
    124: "bitmap_v5_header",
}
Compressions = (
    "rgb",
    "rle8",
    "rle4",
    "bitfields",
    "jpeg",
    "png",
    "alpha_bitfields"
)

def parser(file):
    if type(file) is str:
        name = file
        with open(name, "rb") as file: parser(file)
        return

    magic = file.read(2)
    assert magic == b"BM"

    (f_size, rez1, rez2, bm_offset, h_size, W, H, planes, bitperpixel,
        compression, sizeimage, x_res, y_res, clr_used, clr_imp) = f_unpack(file, "<IHHIIiiHHIIIIII")
    print(f"f_size: {f_size}")
    print(f"size: {W}x{H}")
    print(f"bits per pixel: {bitperpixel}")
    print(f"resolution: {x_res}x{y_res}")
    print(f"colors: {clr_used} imp: {clr_imp}")

    real_size = os.stat(file.name).st_size

    assert f_size == real_size
    assert rez1 == rez2 == 0
    assert W > 0 and H > 0

    if bitperpixel in (4, 8):
        colors = 1 << bitperpixel
        palette = f_unpack(file, f"<{colors}I")
        print(tuple(hex(color)[2:-2].rjust(6, '0') for color in palette))
    # else: palette = None

    assert file.tell() == bm_offset

    W_padded = (W + 3) // 4 * 4 # 0 = 0 | 1,2,3,4 = 4 | 5,6,7,8 = 8...
    left = f_size - file.tell()
    bpp = left / (W_padded * H)
    print("bytes per pixel (до конца файла):", bpp)
    print("лишние байты:", left - W_padded * H * bpp)

    header_type = HeaderTypes[h_size]
    compression = Compressions[compression]
    print("header type:", header_type)
    print("compression:", compression)

    assert header_type == "bitmap_info_header", "Остальные типы пока не поддерживаются"
    assert compression == "rgb", "Остальные компрессии пока не поддерживаются"

def filter(color):
    R = color & 255
    G = color >> 8 & 255
    B = color >> 16 & 255
    gray = round(R * 0.3 + G * 0.59 + B * 0.11)
    return gray | gray << 8 | gray << 16



def reader(_in, readonly = True, bin_palette = False):
    # header: magic (2b) + file_header (12b) + info_header (40b) = 54b
    assert _in.read(2) == b"BM"
    header = (
        f_size, rez1, rez2, bm_offset, # file_header
        h_size, W, H, planes, bitperpixel, compression, sizeimage, x_res, y_res, clr_used, clr_imp # info_header
    ) = unpack("<IHHIIiiHHIIIIII", _in.read(52))

    real_size = os.stat(_in.name).st_size

    assert f_size == real_size
    assert HeaderTypes[h_size] == "bitmap_info_header", "Остальные типы пока не поддерживаются"
    assert Compressions[compression] == "rgb", "Остальные компрессии пока не поддерживаются"
    assert bitperpixel in (8, 24, 4)

    if bitperpixel in (4, 8):
        colors = 1 << bitperpixel
        palette = tuple(_in.read(4)[:3] for i in range(colors)) if bin_palette else f_unpack(_in, f"<{colors}I")
    else: palette = None

    assert _in.tell() == bm_offset

    if readonly: matrix = _in.read()
    else:
        matrix = bytearray(real_size - _in.tell())
        _in.readinto(matrix)

    return header, palette, matrix



def lab_1(_in, out):
    assert _in.read(2) == b"BM"

    header = _in.read(52)
    (f_size, rez1, rez2, bm_offset, h_size, W, H, planes, bitperpixel,
        compression, sizeimage, x_res, y_res, clr_used, clr_imp) = unpack("<IHHIIiiHHIIIIII", header)

    assert bitperpixel == 8

    palette = f_unpack(_in, "<256I")
    matrix = _in.read()

    # palette = [randint(0, 0xffffff) for i in range(256)]
    palette = map(filter, palette)

    out.write(b"BM")
    out.write(header)
    out.write(pack("<256I", *palette))
    out.write(matrix)



def lab_2(_in, out):
    header, palette, matrix = reader(_in, False)
    f_size, rez1, rez2, bm_offset, h_size, W, H, planes, bitperpixel, compression, sizeimage, x_res, y_res, clr_used, clr_imp = header

    assert bitperpixel in (8, 24)

    W_padded = (W + 3) // 4 * 4 # 0 = 0 | 1,2,3,4 = 4 | 5,6,7,8 = 8...

    clr_used_m1 = clr_used - 1
    if bitperpixel == 8:
        for y in chain(range(15), range(H - 15, H)):
            yW = y * W_padded
            for x in range(W):
                matrix[x + yW] = randint(0, clr_used_m1)
        for y in range(15, H - 15):
            yW = y * W_padded
            for x in chain(range(15), range(W - 15, W)):
                matrix[x + yW] = randint(0, clr_used_m1)
    else:
        for y in chain(range(15), range(H - 15, H)):
            yW = y * W_padded
            for x in range(W):
                pos = (x + yW) * 3
                for i in range(pos, pos + 3): matrix[i] = randint(0, 255)
        for y in range(15, H - 15):
            yW = y * W_padded
            for x in chain(range(15), range(W - 15, W)):
                pos = (x + yW) * 3
                for i in range(pos, pos + 3): matrix[i] = randint(0, 255)

    # W = -W # так нельзя
    # H = -H # а так можно. Перевернёт кота
    # W += 1 # секретный последний столбик пикселей в CAT256.bmp (стенография?) (понял, padding-байты)
    # W = W_pad

    out.write(b"BM")
    out.write(pack("<IHHIIiiHHIIIIII", f_size, rez1, rez2, bm_offset, h_size, W, H, planes, bitperpixel, compression, sizeimage, x_res, y_res, clr_used, clr_imp))
    if palette: out.write(pack("<256I", *palette))
    out.write(matrix)



def lab_3(_in, out):
    header, palette, matrix = reader(_in)
    f_size, rez1, rez2, bm_offset, h_size, W, H, planes, bitperpixel, compression, sizeimage, x_res, y_res, clr_used, clr_imp = header

    assert bitperpixel in (8, 24)

    W_padded = (W + 3) // 4 * 4 # 0 = 0 | 1,2,3,4 = 4 | 5,6,7,8 = 8...
    H_pad = b"\0" * ((H + 3) // 4 * 4 - H)

    if bitperpixel == 8:
        int2bytes = tuple(bytes((i,)) for i in range(256))
        get = lambda x, y: int2bytes[matrix[x + y * W_padded]]
    else:
        def get(x, y):
            pos = (x + y * W_padded) * 3
            return matrix[pos : pos + 3]



    write = out.write

    write(b"BM")
    write(pack("<IHHIIiiHHIIIIII", f_size, rez1, rez2, bm_offset, h_size, H, W, planes, bitperpixel, compression, sizeimage, x_res, y_res, clr_used, clr_imp))
    if palette: write(pack("<256I", *palette))

    for x in range(W -1, -1, -1): #  reverse здесь - по часовой
        for y in range(H):  #  reverse здесь - против часовой
            write(get(x, y))
        write(H_pad)

    out.write(matrix)



def lab_4(_in, canvas_maker):
    header, palette, matrix = reader(_in, bin_palette = True)
    f_size, rez1, rez2, bm_offset, h_size, W, H, planes, bitperpixel, compression, sizeimage, x_res, y_res, clr_used, clr_imp = header

    assert bitperpixel in (4, 8, 24)

    W_padded = (W + 3) // 4 * 4 # 0 = 0 | 1,2,3,4 = 4 | 5,6,7,8 = 8...
    put_pixel, ready = canvas_maker(W, H)

    if bitperpixel == 4:
        W_padded //= 2
        for y in range(H):
            yW = y * W_padded
            for x in range(0, W, 2):
                color = matrix[yW + (x >> 1)]
                # put_pixel(x, y, palette[color >> 4])
                # put_pixel(x + 1, y, palette[color & 15])
                put_pixel(x + 1, y, palette[color >> 4]) # специально поменял +1 для solve4_2.png
                put_pixel(x, y, palette[color & 15])
    elif bitperpixel == 8:
        for y in range(H):
            yW = y * W_padded
            for x in range(W):
                put_pixel(x, y, palette[matrix[yW + x]])
    else:
        for y in range(H):
            yW = y * W_padded
            for x in range(W):
                pos = (yW + x) * 3
                put_pixel(x, y, matrix[pos : pos + 3])
    ready()



def lab_5(_in, out, scale):
    header, palette, matrix = reader(_in)
    f_size, rez1, rez2, bm_offset, h_size, W, H, planes, bitperpixel, compression, sizeimage, x_res, y_res, clr_used, clr_imp = header

    assert bitperpixel == 8

    W_padded = (W + 3) // 4 * 4 # 0 = 0 | 1,2,3,4 = 4 | 5,6,7,8 = 8...
    new_W = round(W * scale)
    new_H = round(H * scale)
    W_pad = b"\0" * ((new_W + 3) // 4 * 4 - new_W)

    int2bytes = tuple(bytes((i,)) for i in range(256))
    get = lambda x, y: int2bytes[matrix[x + y * W_padded]]



    write = out.write

    write(b"BM")
    write(pack("<IHHIIiiHHIIIIII", f_size, rez1, rez2, bm_offset, h_size, new_W, new_H, planes, bitperpixel, compression, sizeimage, x_res, y_res, clr_used, clr_imp))
    if palette: write(pack("<256I", *palette))

    inv_scale = 1 / scale
    for y in range(new_H):
        y2 = int(y * inv_scale)
        for x in range(new_W):
            x2 = int(x * inv_scale)
            write(get(x2, y2))
        write(W_pad)

    out.write(matrix)





in_name = os.path.join("orig", "CAT256.BMP")
in_name2 = os.path.join("orig", "_сarib_TC.bmp")
in_name3 = os.path.join("orig", "CAT16.bmp")

parser(in_name3)

def solve_1():
    with open(in_name, "rb") as _in:
        with open("gray_cat.bmp", "wb") as out:
            lab_1(_in, out)

def solve_2():
    with open(in_name, "rb") as _in:
        with open("border.bmp", "wb") as out:
            lab_2(_in, out)
    with open(in_name2, "rb") as _in:
        with open("border2.bmp", "wb") as out:
            lab_2(_in, out)

def solve_3():
    with open(in_name, "rb") as _in:
        with open("rotate.bmp", "wb") as out:
            lab_3(_in, out)
    with open(in_name2, "rb") as _in:
        with open("rotate2.bmp", "wb") as out:
            lab_3(_in, out)

def solve_4():
    from tkinter import Tk, Canvas
    from PIL import Image, ImageTk # pip install Pillow

    root_W = root_H = 0
    border = 2
    def canvas_maker(W, H):
        nonlocal root_W, root_H
        root_W += W + border * 2
        root_H = max(root_H, H + border * 2)

        canvas = Canvas(root, bg = "black", width = W, height = H)
        canvas.grid(row = 0, column = len(anti_gc))

        H_m1 = H - 1
        matrix = np.zeros((H, W, 3), dtype=np.uint8)
        def put_pixel(x, y, color):
            r, g, b = color
            matrix[H_m1 - y, x] = b, g, r
        def ready():
            img = ImageTk.PhotoImage(image = Image.fromarray(matrix))
            canvas.create_image((W / 2 + border - 0.5, H / 2 + border - 0.5), image=img, state="normal")
            anti_gc.append(img) # иначе мусоросборщик (gc) съест все PhotoImage, кроме последней из-за перезаписей 'img' перменной
        return put_pixel, ready

    root = Tk()
    root.title("VectorASD")

    anti_gc = []

    for name in (in_name, in_name2, in_name3):
        T = time()
        with open(name, "rb") as _in:
            lab_4(_in, canvas_maker)
        td = time() - T
        print(f"Время загрузки {name !r}: {round(td, 3)} sec.")

    root.geometry(f"{root_W}x{root_H}")
    root.mainloop()

def solve_5():
    with open(in_name, "rb") as _in:
        for i, scale in enumerate((0.1, 0.25, 0.5, 1, 2, 4, 10)):
            _in.seek(0)
            with open(f"scale{i}_{scale}.bmp", "wb") as out:
                lab_5(_in, out, scale)

solve_1()
solve_2()
solve_3()
# solve_4()
solve_5()
