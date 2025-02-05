from lab8 import PCX_parser, cat256, _200001
from struct import pack

# 17 по журналу, значит 7-ой вариант

# огромная статья про сами октодеревья: https://habr.com/ru/articles/334990/
# здесь описан общий случай их использования для коллизий, но с RGB-кубом всё ЕЩЁ в 1000 раз проще

class OctreeNode:
    def __init__(self):
        self.r = self.g = self.b = self.count = 0
        self.children = [None] * 8
    def add(self, r, g, b, depth = 7):
        self.r += r
        self.g += g
        self.b += b
        self.count += 1 # ускоритель для нахождения среднего арифметического УЖЕ потом, а не с каждый последующий пикселем

        if depth < 0: return # естественный ограничитель True Color, ведь битов больше нет

        index = (r >> depth & 1) << 2 | (g >> depth & 1) << 1  | (b >> depth & 1)
        children = self.children
        node = children[index]
        if node is None: node = children[index] = OctreeNode()
        node.add(r, g, b, depth - 1)
    def color(self):
        count = self.count
        return self.r // count, self.g // count, self.b // count

def generator(root, limit):
    assert limit >= 1

    palette = set() # защита от повторов
    add = palette.add

    queue2 = [root] # по скольку нам необходим перебор в ширину, а не в глубину, обычной рекурсии будет маловато

    while queue2:
        queue = sorted(queue2, key = lambda node: node.count, reverse = True) # военная хитрость. "x1000" к качеству картинке
        # print(tuple(node.count for node in queue)) действительно по убыванию
        # ☣☣☣ самый простой способ испортить Октодерево - поменять reverse на False, т.е. вообще его убрать. В палитру попадут редчайшие цвета ☣☣☣
        queue2 = []
        append = queue2.append
        for node in queue:
            color = node.color()
            if color in palette: continue
            add(color)
            if len(palette) >= limit:
                queue2.clear()
                break # тормоза

            for child in node.children:
                if child is not None: append(child)

    return tuple(palette)

def rgr(_in, out):
    W, H, bits_per_pixel, matrix, EGA_palette, VGA_palette = PCX_parser(_in)

    assert bits_per_pixel == 8
    assert EGA_palette == (b"\0\0\0",) * 16 # смысл в ней, если есть VGA? Потому здесь нули

    root = OctreeNode()
    add = root.add
    for y in range(H):
        for x in range(W):
            add(*VGA_palette[matrix[y][x]])

    print("avg:", "#" + bytes(root.color()).hex(), generator(root, 1) == (root.color(),))
    print("max:", len(generator(root, 1 << 24)))
    # cat256: avg = #6a7154; max = 320
    # _200001: avg = #525439; max = 328
    # особенность октанового дерева: на выходе может быть больше цветов, чем вообще было в палитре изначально

    palette = generator(root, 16)
    # palette = ((255, 0, 0),) * 16 # синий. Это объясняет все предыдущие лабораторные, почему GBR, а не RGB
    clr_used = clr_imp = len(palette)
    assert clr_used <= 16

    VGA_filtered = tuple(
        min(
            (
                # (r - r2) ** 2 + (g - g2) ** 2 + (b - b2) ** 2, худший
                # 0.3 * (r - r2) ** 2 + 0.59 * (g - g2) ** 2 + 0.11 * (b - b2) ** 2,
                (0.3 * (r - r2)) ** 2 + (0.59 * (g - g2)) ** 2 + (0.11 * (b - b2)) ** 2, # лучший
                i
            )
            for i, (r, g, b) in enumerate(palette)
        )[1]
        for r2, g2, b2 in VGA_palette
    ) # а теперь представьте, что изначально это всё было записано в одну строчку... прямо как в сложных регулярных выражениях...
    print(VGA_filtered)

    palette = tuple(bytes(color)[::-1] for color in palette) + (b"\0\0\0",) * (16 - clr_used) # здесь же флип с rgb в gbr
    assert len(palette) == 16

    W_padded = (W + 3) // 4 * 4 # 0 = 0 | 1,2,3,4 = 4 | 5,6,7,8 = 8...
    W_padded //= 2 # по скольку картинка 4-битная
    rez1 = rez2 = x_res = y_res = 0
    h_size = 40 # bitmap_info_header
    planes = 1
    bitperpixel = 4
    compression = 0 # rgb
    bm_offset = 2 + 12 + h_size + 16 * 4
    sizeimage = W_padded * H
    f_size = bm_offset + sizeimage
    print("f_size:", f_size)



    write = out.write

    write(b"BM")
    write(pack("<IHHIIiiHHIIIIII", f_size, rez1, rez2, bm_offset, h_size, W, H, planes, bitperpixel, compression, sizeimage, x_res, y_res, clr_used, clr_imp))
    for color in palette: write(color + b"\0")

    for y in range(H -1, -1, -1): # перевёрнутый Y
        buffer = bytearray(W_padded) # гениально! идея с буфферами из PCX автоматически!!! создаёт те самые дополнительные нули в конце
        pos = 0
        row = matrix[y]
        for x in range(0, W, 2):
            color = VGA_filtered[row[x]]
            color2 = VGA_filtered[row[x + 1]]
            buffer[pos] = color << 4 | color2
            pos += 1
        write(buffer)



def solve():
    with open(cat256, "rb") as _in:
        with open("rgr_cat16.bmp", "wb") as out:
            rgr(_in, out)

    with open(_200001, "rb") as _in:
        with open("rgr_tiger16.bmp", "wb") as out:
            rgr(_in, out)

solve()
