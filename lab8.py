from struct import pack, unpack, calcsize
import os
from time import time

# https://formats.kaitai.io/pcx/index.html
#    ZSoft PCX File Format Technical Reference Manual:
# https://web.archive.org/web/20100206055706/http://www.qzx.com/pc-gpe/pcx.txt
#    Дополнительно (PCX File Format Summary):
# https://gist.github.com/warmwaffles/64d1a8c677997fc8c3df4fa7364d33cc

Versions = {
    0: "v2_5",
    2: "v2_8_with_palette",
    3: "v2_8_without_palette",
    4: "paintbrush_for_windows",
    5: "v3_0",
}
Encodings = {
    1: "rle",
}

def f_unpack(file, format):
    L = calcsize(format)
    return unpack(format, file.read(L))

def EGA_64_standart_color(ega): # https://moddingwiki.shikadi.net/wiki/EGA_Palette
    red = 85 * (ega >> 1 & 2 | ega >> 5 & 1)
    green = 85 * (ega & 2 | ega >> 4 & 1)
    blue = 85 * (ega << 1 & 2 | ega >> 3 & 1)
    return red, green, blue
def EGA_16_standart_color(ega): # сам синтезировал (упростил) из их EGA-64
    bright = ega >> 3 & 1
    red = 85 * (ega >> 1 & 2 | bright)
    green = 85 * (ega & 2 | bright)
    blue = 85 * (ega << 1 & 2 | bright)
    return red, green, blue

EGA_64_standart_palette = tuple(EGA_64_standart_color(i) for i in range(64))
EGA_16_standart_palette = tuple(EGA_16_standart_color(i) for i in range(16))
# print(EGA_64_standart_palette)
# print(EGA_16_standart_palette)



def PCX_parser(_in):
    assert _in.read(1) == b"\n", "file-magic error"
    (version, encoding, bits_per_pixel, img_x_min, img_y_min,
        img_x_max, img_y_max, hdpi, vdpi) = f_unpack(_in, "<3B6H")
    EGA_palette = tuple(_in.read(3) for i in range(16)) # в некоторых версиях она не поддерживается. Вместо неё EGA_16_standart_palette, наверное
    reserved, num_planes, bytes_per_line, palette_info, h_screen_size, b_screen_size = f_unpack(_in, "<2B4H")

    assert _in.read(128 - 74) == b"\0" * (128 - 74) # reserved
    assert reserved == 0

    version = Versions[version]
    encoding = Encodings[encoding]
    print("версия:", version)
    print("кодировка:", encoding)
    assert version == "v3_0", "Пока не встречал PCX в других версиях"
    assert encoding == "rle" # а других ZSoft и не завёз по сей день, ибо уже давно поздно этот формат продвигать дальше

    width = img_x_max - img_x_min + 1 # XSIZE в референсе
    height = img_y_max - img_y_min + 1 # YSIZE в референсе
    print(f"size: {width}x{height}")
    total_bytes = num_planes * bytes_per_line # одна полная несжатая строка сканирования
    # основная его задача: заранее выделить total_bytes на строчку ПЕРЕД декодированием
    # справа могут быть заполнители, как и в BMP, но здесь уже требование чётности, а не деления на 4

    # референс описывает, как именно нужно интерпретировать rle-кодировку ;'-}
    # вот только странность в том, что описана ситуация с 11 и 00 старшими битами, а вот про 01 и 10 забыли, что на практике равнозначно 00

    matrix = [None] * height
    for line in range(height):
        buffer = bytearray(total_bytes)
        pos = 0
        read = _in.read
        while pos < total_bytes:
            b = read(1)[0]
            if b >> 6 == 3:
                value = read(1)[0]
                for i in range(b & 0x3f):
                    buffer[pos] = value
                    pos += 1
            else:
                buffer[pos] = b
                pos += 1
        matrix[line] = buffer

    if version == "v3_0" and bits_per_pixel == 8 and num_planes == 1: # точное условие наличия VGA-палитры
        b = _in.read(1)
        while b == b"\0": b = _in.read(1) # странный padding
        assert _in.tell() % 8 == 0 # видимо для того, чтобы следующий байт после magic делился на 8 нацело
        assert b == b"\x0c", "VGA-palette-magic error"
        VGA_palette = tuple(_in.read(3) for i in range(256))
    else: VGA_palette = None

    f_size = os.stat(_in.name).st_size
    assert _in.tell() == f_size

    return width, height, bits_per_pixel, matrix, EGA_palette, VGA_palette



def lab_8(_in, canvas_maker):
    W, H, bits_per_pixel, matrix, EGA_palette, VGA_palette = PCX_parser(_in)

    assert bits_per_pixel == 8
    assert EGA_palette == (b"\0\0\0",) * 16 # смысл в ней, если есть VGA? Потому здесь нули

    put_pixel = canvas_maker(W, H)
    for y in range(H):
        for x in range(W):
            put_pixel(x, y, VGA_palette[matrix[y][x]])



def canvas_printer(solve):
    from tkinter import Tk, Canvas
    from PIL import Image, ImageTk # pip install Pillow (только для преобразования numpy-матрицы в холст!)
    import numpy as np # pip install numpy

    anti_gc = []
    cells = []
    border = 2
    last_ready = None
    def canvas_maker(W, H, columns = 2):
        nonlocal last_ready

        for i in range(10):
            row, column = divmod(i, columns)
            if len(cells) <= row or len(cells[row]) <= column: break
        cell = W + border * 2, H + border * 2
        while len(cells) <= row: cells.append([])
        while len(cells[row]) <= column: cells[row].append(None)
        cells[row][column] = cell
        # print("CELL:", row, column, cells)

        canvas = Canvas(root, bg = "black", width = W, height = H)
        canvas.grid(row = row, column = column)

        H_m1 = H - 1
        matrix = np.zeros((H, W, 3), dtype=np.uint8)
        def put_pixel(x, y, color):
            # r, g, b = color
            # matrix[H_m1 - y, x] = b, g, r
            # в отличии от lab_4, RGB и Y-координата НЕ перевёрнута!!!
            matrix[y, x] = tuple(color)
        def ready():
            img = ImageTk.PhotoImage(image = Image.fromarray(matrix))
            canvas.create_image((W / 2 + border - 0.5, H / 2 + border - 0.5), image=img, state="normal")
            anti_gc.append(img) # иначе мусоросборщик (gc) съест все PhotoImage, кроме последней из-за перезаписей 'img' перменной
        if last_ready is not None: last_ready()
        last_ready = ready
        return put_pixel

    root = None

    def wrap():
        nonlocal root
        root = Tk()
        root.title("VectorASD")

        solve(canvas_maker)
        if last_ready is not None: last_ready()

        # root.geometry(f"{root_W}x{root_H}") оказывается, всё это время по умолчанию оно выставляло нужный мне результат O_o
        root.mainloop()
    return wrap

@canvas_printer
def solve_8(canvas_maker):
    for name in (cat256, _200001):
        T = time()
        with open(name, "rb") as _in:
            lab_8(_in, canvas_maker)
        td = time() - T
        print(f"Время загрузки {name !r}: {round(td, 3)} sec.")





cat256 = os.path.join("orig", "CAT256.PCX")
_200001 = os.path.join("orig", "200001.PCX")

if __name__ == "__main__":
    solve_8()
