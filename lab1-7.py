from struct import unpack, pack, calcsize
import os
from random import randint
from itertools import chain
from time import time
from hashlib import sha256

# https://formats.kaitai.io/bmp/python.html
# https://entropymine.com/jason/bmpsuite/bmpsuite/html/bmpsuite.html

def f_unpack(file, format):
    L = calcsize(format)
    return unpack(format, file.read(L))

# calcsize("<IHHIIiiHHIIIIII") -> 52 (без magic!)

HeaderTypes = { # без учёта magic и file_header
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
    B = color & 255
    G = color >> 8 & 255
    R = color >> 16 & 255
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



def lab_6(_in, watermark, out, settings):
    def watermark_getter():
        # watermark
        header, palette, matrix = reader(watermark, bin_palette = True)
        _, _, _, _, _, W, H, _, bitperpixel, _, _, _, _, _, _ = header

        assert bitperpixel == 8

        scale_x = W / w_width
        scale_y = H / w_height
        W_padded = (W + 3) // 4 * 4 # 0 = 0 | 1,2,3,4 = 4 | 5,6,7,8 = 8...

        int2bytes = tuple(bytes((i,)) for i in range(256))
        return scale_x, scale_y, lambda x, y: palette[matrix[x + y * W_padded]]

    # main
    header, _, matrix = reader(_in)
    _, _, _, _, _, W, H, _, bitperpixel, _, _, _, _, _, _ = header

    assert bitperpixel == 24

    margin, w_width, w_height, up, left, alpha = settings
    left = margin if left else W - w_width - margin
    top = H - w_height - margin if up else margin

    scale_x, scale_y, get = watermark_getter()



    write = out.write
    write(b"BM")
    write(pack("<IHHIIiiHHIIIIII", *header))

    W_padded_2 = (W + 3) // 4 * 4
    W_pad = b"\0" * (W_padded_2 - W)
    inv_alpha = 1 - alpha

    for y in range(H):
        yW = y * W_padded_2
        for x in range(W):
            pos = (yW + x) * 3
            color = matrix[pos : pos + 3]
            if x >= left and y >= top:
                wx, wy = x - left, y - top
                if wx < w_width and wy < w_height:
                    r, g, b = color
                    r2, g2, b2 = get(int(wx * scale_x), int(wy * scale_y))
                    color = bytes((int(r * inv_alpha + r2 * alpha), int(g * inv_alpha + g2 * alpha), int(b * inv_alpha + b2 * alpha)))
            write(color)
            #print(x, y, "|", get(x, y))
        write(W_pad)



def reader_2b(file, matrix_size):
    L = os.stat(file.name).st_size
    pad_size = matrix_size // 4 - L - 37
    assert pad_size >= 0 # чтобы гарантировать, что всё влезет, либо оно вообще не примется за работу

    digester = sha256()
    for i in range(L):
        num, = byte = file.read(1)
        digester.update(byte)
        yield num >> 6
        yield num >> 4 & 3
        yield num >> 2 & 3
        yield num & 3
    # pad = b"\0" * pad_size больше палится, если все - нули
    pad = bytes(randint(0, 255) for i in range(pad_size))
    append = b"".join((pad, digester.digest(), pack("<I", L), b"\1"))
    for num in append:
        yield num >> 6
        yield num >> 4 & 3
        yield num >> 2 & 3
        yield num & 3

def reader_4b(file, matrix_size):
    L = os.stat(file.name).st_size
    pad_size = matrix_size // 2 - L - 37
    assert pad_size >= 0

    digester = sha256()
    for i in range(L):
        num, = byte = file.read(1)
        digester.update(byte)
        yield num >> 4 & 15
        yield num & 15
    pad = bytes(randint(0, 255) for i in range(pad_size))
    append = b"".join((pad, digester.digest(), pack("<I", L), b"\2"))
    for num in append:
        yield num >> 4 & 15
        yield num & 15

def reader_6b(file, matrix_size):
    L = os.stat(file.name).st_size
    pad_size = matrix_size * 3 // 4 - L - 37
    assert pad_size >= 0

    digester = sha256()
    for i in range((L + 2) // 3):
        bytez = file.read(3)
        try: b1, b2, b3 = bytez
        except ValueError:
            b1, b2, b3 = bytez[0], (bytez[1] if len(bytez) == 2 else 0), 0
        digester.update(bytez)
        num = b1 << 16 | b2 << 8 | b3
        yield num >> 18
        yield num >> 12 & 63
        yield num >> 6 & 63
        yield num & 63
    pad = bytes(randint(0, 255) for i in range(pad_size))
    append = b"".join((pad, digester.digest(), pack("<I", L), b"\3"))
    for i in range(0, len(append) + 2, 3):
        bytez = append[i : i + 3]
        try: b1, b2, b3 = bytez
        except ValueError:
            b1, b2, b3 = bytez[0], (bytez[1] if len(bytez) == 2 else 0), 0
        num = b1 << 16 | b2 << 8 | b3
        yield num >> 18
        yield num >> 12 & 63
        yield num >> 6 & 63
        yield num & 63

def lab_7(_in, out, file, mode):
    #   структура файла (size b.):
    # полезная нагрузка (L b.)
    # паддинг (size-L-37 b.)
    # sha256 (32 b.)
    # размер полезной нагрузки (4 b.)
    # режим (1 b.)

    header, palette, matrix = reader(_in)
    f_size, rez1, rez2, bm_offset, h_size, W, H, planes, bitperpixel, compression, sizeimage, x_res, y_res, clr_used, clr_imp = header

    assert bitperpixel == 24
    assert mode in (2, 4, 6)

    if mode == 2:
        # print(len(bytes(bit_reader))) # равно len(matrix), т.е. 1440000
        # print(max(bytes(bit_reader))) # равно 3, т.к. это максималка для двух битов
        bit_reader = iter(reader_2b(file, len(matrix)))
        matrix = bytes((byte & 0xfc | next(bit_reader) for byte in matrix))
    elif mode == 4:
        bit_reader = iter(reader_4b(file, len(matrix)))
        matrix = bytes((byte & 0xf0 | next(bit_reader) for byte in matrix))
    else: # mode == 6
        bit_reader = iter(reader_6b(file, len(matrix)))
        matrix = bytes((byte & 0xc0 | next(bit_reader) for byte in matrix))

    write = out.write
    write(b"BM")
    write(pack("<IHHIIiiHHIIIIII", *header))
    write(matrix)



def unpacker_2b(matrix):
    L = len(matrix) // 4
    mat = iter(matrix)
    result = bytearray(L)
    for i in range(L):
        b1, b2, b3, b4 = next(mat), next(mat), next(mat), next(mat)
        result[i] = (b1 & 3) << 6 | (b2 & 3) << 4 | (b3 & 3) << 2 | b4 & 3
    return result

def unpacker_4b(matrix):
    L = len(matrix) // 2
    mat = iter(matrix)
    result = bytearray(L)
    for i in range(L):
        b1, b2 = next(mat), next(mat)
        result[i] = (b1 & 15) << 4 | b2 & 15
    return result

def unpacker_6b(matrix):
    L = len(matrix) * 3 // 4
    mat = iter(matrix)
    result = bytearray(L)
    for i in range(0, L, 3):
        b1, b2, b3, b4 = next(mat), next(mat), next(mat), next(mat)
        num = (b1 & 63) << 18 | (b2 & 63) << 12 | (b3 & 63) << 6 | b4 & 63
        result[i] = num >> 16
        result[i + 1] = num >> 8 & 255
        result[i + 2] = num & 255
    return result

def lab_7_2(_in, out):
    header, palette, matrix = reader(_in)
    f_size, rez1, rez2, bm_offset, h_size, W, H, planes, bitperpixel, compression, sizeimage, x_res, y_res, clr_used, clr_imp = header

    shift = len(matrix) - len(matrix) // 4 * 4
    assert bitperpixel == 24
    mode = matrix[-1 - shift] & 3
    assert mode

    file = (None, unpacker_2b, unpacker_4b, unpacker_6b)[mode](matrix)

    hash = file[-37:-5]
    size = unpack("<I", file[-5:-1])[0]
    assert file[-1] == mode

    data = file[:size]
    assert sha256(data).digest() == hash, "Неверный sha256-digest"
    out.write(data)





def solve_1():
    with open(cat256, "rb") as _in:
        with open("gray_cat.bmp", "wb") as out:
            lab_1(_in, out)

def solve_2():
    with open(cat256, "rb") as _in:
        with open("border.bmp", "wb") as out:
            lab_2(_in, out)
    with open(carib, "rb") as _in:
        with open("border2.bmp", "wb") as out:
            lab_2(_in, out)

def solve_3():
    with open(cat256, "rb") as _in:
        with open("rotate.bmp", "wb") as out:
            lab_3(_in, out)
    with open(carib, "rb") as _in:
        with open("rotate2.bmp", "wb") as out:
            lab_3(_in, out)

def solve_4():
    from tkinter import Tk, Canvas
    from PIL import Image, ImageTk # pip install Pillow (только для преобразования numpy-матрицы в холст!)
    import numpy as np # pip install numpy

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

    for name in (cat256, carib, cat16):
        T = time()
        with open(name, "rb") as _in:
            lab_4(_in, canvas_maker)
        td = time() - T
        print(f"Время загрузки {name !r}: {round(td, 3)} sec.")

    root.geometry(f"{root_W}x{root_H}")
    root.mainloop()

def solve_5():
    with open(cat256, "rb") as _in:
        for i, scale in enumerate((0.1, 0.25, 0.5, 1, 2, 4, 10)):
            _in.seek(0)
            with open(f"scale{i}_{scale}.bmp", "wb") as out:
                lab_5(_in, out, scale)

def solve_6():
    margin = 8
    width = height = 256
    alpha = 0.6 # непрозрачность именно водяного знака
    up = False
    left = False
    settings  = margin, width, height, up, left, alpha

    with open(carib, "rb") as _in:
        with open(cat256, "rb") as watermark:
            with open("watermarked.bmp", "wb") as out:
                lab_6(_in, watermark, out, settings)
        _in.seek(0)

        settings = margin, width, height, False, True, alpha
        with open(favicon, "rb") as watermark:
            with open("watermarked2.bmp", "wb") as out:
                lab_6(_in, watermark, out, settings)
        _in.seek(0)

        settings = margin, width, height, True, False, alpha
        with open(favicon2, "rb") as watermark:
            with open("watermarked3.bmp", "wb") as out:
                lab_6(_in, watermark, out, settings)

def solve_7():
    # внутри carib 800x600x3 байтов, т.е. 1440000. 1440000 / 1024**2 = 1.37 Мб, что и подтверждает win10:explorer.exe
    # ещё и фактический размер файла 1440054, из которых 54 байта: magic (2b) + file_header (12b) + info_header (40b)
    # 25%: 360000 байтов
    # 50%: 720000 байтов
    # 75%: 1080000 байтов

    with open(carib, "rb") as _in:
        with open("steganography_2.bmp", "wb") as out:
            with open(secret_360k, "rb") as file:
                lab_7(_in, out, file, 2)
        _in.seek(0)

        with open("steganography_4.bmp", "wb") as out:
            with open(secret_720k, "rb") as file:
                lab_7(_in, out, file, 4)
        _in.seek(0)

        with open("steganography_6.bmp", "wb") as out:
            with open(secret_1080k, "rb") as file:
                lab_7(_in, out, file, 6)

def solve_7_2():
    names = secret_360k, secret_720k, secret_1080k
    for i in (2, 4, 6):
        with open(f"steganography_{i}.bmp", "rb") as _in:
            with open(names[i // 2 - 1], "wb") as out:
                lab_7_2(_in, out)





cat256 = os.path.join("orig", "CAT256.BMP")
carib = os.path.join("orig", "_сarib_TC.bmp")
cat16 = os.path.join("orig", "CAT16.bmp")
favicon = os.path.join("orig", "favicon.bmp")
favicon2 = os.path.join("orig", "favicon2.bmp")
secret_360k = os.path.join("orig", "steganography_360k.7z")
secret_720k = os.path.join("orig", "steganography_720k.7z")
secret_1080k = os.path.join("orig", "steganography_1080k.7z")

parser(favicon)

# solve_1()
# solve_2()
# solve_3()
# solve_4()
# solve_5()
# solve_6()
# solve_7()
solve_7_2()

size = 89527 - 817 # для 360k
size = 46247 - 800 - 15 # для 370k
# with open(r"D:\Meow\Desktop\Учёба\ПГИ\orig\random", "wb") as file: file.write(bytes(randint(0, 255) for i in range(size)))
