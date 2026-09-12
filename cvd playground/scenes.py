from PIL import Image, ImageDraw, ImageFont
from typing import Callable, Optional
from scipy.spatial import Voronoi
from matplotlib.patches import Polygon
from tqdm import tqdm
from cvd import *
import re


def binary_text(text: str, h: int, w: int, font_path: Optional[str] = None) -> NDArray:
    image = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(image)

    def load_font(size: int) -> ImageFont:
        if font_path is None:
            return ImageFont.load_default(size=size)
        return ImageFont.truetype(font_path, size=size)

    size = 1
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, load_font(size))
    while x1 - x0 < w * 0.8 and y1 - y0 < h * 0.8:
        size *= 2
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, load_font(size))
    while x1 - x0 > w * 0.8 or y1 - y0 > h * 0.8:
        size -= 1
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, load_font(size))

    draw.text((w // 2, h // 2), text, fill=1, font=load_font(size), spacing=0, anchor='mm',
              align='center')
    return np.array(image) > 0


def generate_voronoi(mask: NDArray, cvd_type: str, rgb: NDArray,
                     severity: float = 0.5, n: int = 5000) -> None:
    h, w = mask.shape

    start, end = confusion_line(rgb / 255, cvd_type)
    if np.random.rand() > 0.5:
        start, end = end, start
    a = np.random.rand() * (1 - severity)
    b = a + severity
    c1 = lms_to_rgb(start * (1 - a) + end * a)
    c2 = lms_to_rgb(start * (1 - b) + end * b)

    points = np.random.rand(n, 2) * np.array([h, w])[None]
    vor = Voronoi(points)

    rand = np.random.randn(n + 1, 3) / 10 + 1
    fig, ax = plt.subplots(figsize=(w / h * 7, 7))
    for i, region in enumerate(vor.regions):
        if -1 not in region and len(region) > 0:
            polygon = np.array([vor.vertices[i] for i in region])
            # polygon[:, 1] = h - polygon[:, 1]
            y, x = np.round(np.mean(polygon, axis=0)).astype(int)
            if mask[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)]:
                color = np.clip(c1 * rand[i], 0, 1)
            else:
                color = np.clip(c2 * rand[i], 0, 1)
            # ax.fill(*zip(*polygon), color=color, edgecolor='white', linewidth=1)
            polygon[:, 0] = h - polygon[:, 0]
            ax.add_patch(Polygon(np.flip(polygon, axis=1), closed=True, facecolor=color))

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)


def generate_grid_inverse(mask: NDArray, cvd_type: str, n: int = 10) -> None:
    h, w = mask.shape
    data = np.empty((h * n, w * n, 3))

    # a = np.add.outer(np.arange(n), np.arange(n), dtype=float)
    # a /= np.max(a)
    # a = a.flatten()

    for i in tqdm(range(h * w)):
        x, y = divmod(i, h)
        start, end = confusion_line(np.random.rand(3), cvd_type)
        if mask[y, x]:
            start = rgb_to_lms(np.random.rand(3))
            end = rgb_to_lms(np.random.rand(3))
            diff = np.abs(start - end)
            c1 = np.mean(lms_to_rgb(start))
            c2 = np.mean(lms_to_rgb(end))
            rgb_diff = c1 / c2
            if rgb_diff < 1:
                rgb_diff = 1 / rgb_diff

            if cvd_type == 'd':
                condition = lambda diff: (diff[0] < 2 or diff[0] > 20) or diff[1] < 5 or (
                        diff[2] < 0.5 or diff[2] > 1.5) or rgb_diff > 1.5
                # condition = lambda diff: (diff[0] < 3 or diff[0] > 20) or diff[1] > 10 or (diff[2] < 0.5 or diff[2] > 1.5)
            elif cvd_type == 'p':
                # condition = lambda diff: diff[0] > 10 or (diff[1] < 2 or diff[1] > 10) or (diff[2] < 0.5 or diff[2] > 1.5)
                condition = lambda diff: diff[0] < 5 or (diff[1] < 2 or diff[1] > 10) or (
                        diff[2] < 0.5 or diff[2] > 1.5) or rgb_diff > 1.5
            else:
                condition = lambda diff: False

            while condition(diff):
                start = rgb_to_lms(np.random.rand(3))
                end = rgb_to_lms(np.random.rand(3))
                diff = np.abs(start - end)
                c1 = np.mean(lms_to_rgb(start))
                c2 = np.mean(lms_to_rgb(end))
                rgb_diff = c1 / c2
                if rgb_diff < 1:
                    rgb_diff = 1 / rgb_diff

        # a = np.repeat(np.linspace(0, 1, num=n)[None], n, axis=0).flatten()
        # a = np.repeat(np.linspace(0, 1, num=n), n, axis=0)
        a = np.random.rand(n ** 2)
        c = lms_to_rgb(start[None, :] * (1 - a[:, None]) + end[None, :] * a[:, None])
        data[y * n:(y + 1) * n, x * n:(x + 1) * n, :] = c.reshape(n, n, 3)

    plt.imshow(data)

    # t = 10
    # fig, ax = plt.subplots(2, 10)
    # for i in range(t):
    #     start, end, diff = k[i]
    #     start = np.clip(lms_to_linear_rgb(start), 0, 1) ** (1 / 2.4)
    #     end = np.clip(lms_to_linear_rgb(end), 0, 1) ** (1 / 2.4)
    #     ax[0, i].imshow(start [None, None])
    #     ax[1, i].imshow(end[None, None])
    #     ax[1, i].set_title(round(diff))


# https://en.wikipedia.org/wiki/Fourteen-segment_display
# a, b, c, d, e, f, g1, g2, h, i, j, k, l, m
# a, b, c, d, e, f, g, h, i, j, k, l, m, n
CHAR_TO_SEGMENT = {
    '0': 'abcdefkl',
    '1': 'bck',
    '2': 'abghed',
    '3': 'abhcd',
    '4': 'fghbc',
    '5': 'afghcd',
    '6': 'afedghc',
    '7': 'akm',
    '8': 'abcdefgh',
    '9': 'abfghc',
    'A': 'abcefgh',
    'B': 'abcdjmh',
    'C': 'afed',
    'D': 'abcdjm',
    'E': 'afghed',
    'F': 'afghe',
    'G': 'afedch',
    'H': 'feghbc',
    'I': 'ajmd',
    'J': 'edcb',
    'K': 'fegkn',
    'L': 'fed',
    'M': 'feikbc',
    'N': 'feinbc',
    'O': 'abcdef',
    'P': 'abfegh',
    'Q': 'abcdefn',
    'R': 'abfghen',
    'S': 'aihcd',
    'T': 'ajm',
    'U': 'fedbc',
    'V': 'felk',
    'W': 'felnbc',
    'X': 'ikln',
    'Y': 'ikm',
    'Z': 'akld'
}


def generate_segment_inverse(text: str, cvd_type: str) -> None:
    text = text.upper()
    assert all(x in CHAR_TO_SEGMENT for x in text)

    h, w = 200, 140

    def draw_line(x: list[int], y: list[int], colour: NDArray) -> None:
        plt.plot(x, y, color=colour, linewidth=7.5, solid_capstyle='round')

    def fill(segment: str, offset: tuple[int, int], colour: NDArray) -> None:
        y, x = offset
        if segment == 'a':
            draw_line([x + 20, x + w - 20], [y + h - 10, y + h - 10], colour)
        elif segment == 'b':
            draw_line([x + w - 10, x + w - 10], [y + h // 2 + 15, y + h - 25], colour)
        elif segment == 'c':
            draw_line([x + w - 10, x + w - 10], [y + 25, y + h // 2 - 15], colour)
        elif segment == 'd':
            draw_line([x + 20, x + w - 20], [y + 10, y + 10], colour)
        elif segment == 'e':
            draw_line([x + 10, x + 10], [y + 25, y + h // 2 - 15], colour)
        elif segment == 'f':
            draw_line([x + 10, x + 10], [y + h // 2 + 15, y + h - 25], colour)
        elif segment == 'g':
            draw_line([x + 20, x + w // 2 - 10], [y + h // 2, y + h // 2], colour)
        elif segment == 'h':
            draw_line([x + w // 2 + 10, x + w - 20], [y + h // 2, y + h // 2], colour)
        elif segment == 'i':
            draw_line([x + 25, x + w // 2 - 15], [y + h - 25, y + h // 2 + 15], colour)
        elif segment == 'j':
            draw_line([x + w // 2, x + w // 2], [y + h // 2 + 15, y + h - 25], colour)
        elif segment == 'k':
            draw_line([x + w // 2 + 15, x + w - 25], [y + h // 2 + 15, y + h - 25], colour)
        elif segment == 'l':
            draw_line([x + 25, x + w // 2 - 15], [y + 25, y + h // 2 - 15], colour)
        elif segment == 'm':
            draw_line([x + w // 2, x + w // 2], [y + 25, y + h // 2 - 15], colour)
        elif segment == 'n':
            draw_line([x + w // 2 + 15, x + w - 25], [y + h // 2 - 15, y + 25], colour)
        else:
            raise ValueError(segment)

    segments = [chr(x) for x in range(ord('a'), ord('n') + 1)]
    b = 0
    plt.figure(figsize=(1.5 * len(text), 2))
    for i, x in enumerate(text):
        start, end = confusion_line(np.random.rand(3), cvd_type)
        active_segments = set(x for x in CHAR_TO_SEGMENT[x])
        for segment in segments:
            if segment in active_segments:
                # colour = np.array([0, 0, 0])
                b = 1 - b
                a = np.clip(b + np.random.rand() * 0.25, 0, 1)
                colour = lms_to_rgb(start * (1 - a) + end * a)
            else:
                # colour = np.array([0.8, 0.8, 0.8])
                # colour = np.random.rand(3) ** (1 / 2.4)
                comparison = start if np.random.rand() > 0.5 else end
                colour = rgb_to_lms(np.random.rand(3))

                if cvd_type == 'd':
                    condition = lambda diff: (diff[0] < 3 or diff[0] > 20) or diff[1] > 10 or (
                            diff[2] < 0.5 or diff[2] > 1.5)
                elif cvd_type == 'p':
                    condition = lambda diff: diff[0] > 10 or (diff[1] < 2 or diff[1] > 10) or (
                            diff[2] < 0.5 or diff[2] > 1.5)
                else:
                    condition = lambda diff: False
                diff = np.abs(colour - comparison)
                while condition(diff):
                    colour = rgb_to_lms(np.random.rand(3))
                    diff = np.abs(colour - comparison)

                colour = lms_to_rgb(colour)
            fill(segment, (0, i * 160), colour)
    plt.xlim([0, len(text) * 160])
    plt.ylim([0, h])
    plt.axis(False)


def generate_useful_rgb_confusion_line(cvd_type: str) -> tuple[NDArray, NDArray, NDArray]:
    rgb = np.random.rand(3)
    start, end = confusion_line(rgb, cvd_type)
    dist = np.sum((lms_to_rgb(start) - lms_to_rgb(end)) ** 2) ** 0.5
    while dist < 0.5:
        rgb = np.random.rand(3)
        start, end = confusion_line(rgb, cvd_type)
        dist = np.sum((lms_to_rgb(start) - lms_to_rgb(end)) ** 2) ** 0.5
    return rgb, start, end


def saturation(rgb: NDArray) -> float:
    max = np.max(rgb)
    min = np.min(rgb)
    return (max - min) / max


def generate_useful_confusion_line_variants(distractors: list[NDArray]) -> list[list[NDArray]]:
    lines = []
    for rgb in distractors:
        sat0 = saturation(rgb)
        oklab0 = rgb_to_oklab(rgb)
        candidates = []
        done = 0
        while done < 5:
            e = np.array([np.random.rand(), np.random.rand(), 0])
            start, end = confusion_line_custom(rgb, e)
            start_rgb, end_rgb = lms_to_rgb(start), lms_to_rgb(end)

            i = 0
            c = start_rgb * (1 - i) + end_rgb * i
            oklab_diff = np.sum((rgb_to_oklab(c) - oklab0) ** 2) ** 0.5
            while oklab_diff > 0.15:
                i += 0.05
                c = start_rgb * (1 - i) + end_rgb * i
                oklab_diff = np.sum((rgb_to_oklab(c) - oklab0) ** 2) ** 0.5
                if i > 1:
                    break
            if i > 1:
                continue
            print(f'FROM {np.round(c * 255)}, {oklab_diff}')

            j = 1
            c = start_rgb * (1 - j) + end_rgb * j
            oklab_diff = np.sum((rgb_to_oklab(c) - oklab0) ** 2) ** 0.5
            while oklab_diff > 0.15:
                j -= 0.05
                c = start_rgb * (1 - j) + end_rgb * j
                oklab_diff = np.sum((rgb_to_oklab(c) - oklab0) ** 2) ** 0.5
                if j < 0:
                    break
            if j < 0:
                continue
            print(f'TO {np.round(c * 255)}, {oklab_diff}')

            new_start = start * (1 - i) + end * i
            new_end = start * (1 - j) + end * j

            candidates.append([new_start, new_end])
            done += 1
        lines.append(candidates)
    return lines


def generate_ishihara(text: str, cvd_type: str, output_path: str = 'test.svg', mask_size: int = 500) -> None:
    rgb, start, end = generate_useful_rgb_confusion_line(cvd_type)

    n = 10
    start_rgb = lms_to_rgb(start)
    end_rgb = lms_to_rgb(end)
    distractors = [start_rgb * x / n + end_rgb * (n - x) / n for x in range(n + 1)]
    lines = generate_useful_confusion_line_variants(distractors)

    mask = binary_text(text, mask_size, mask_size)

    def drawer(x: float, y: float) -> str:
        xx = np.clip(int(x * mask_size), 0, mask_size - 1)
        yy = np.clip(int(y * mask_size), 0, mask_size - 1)

        t = np.random.rand()
        if mask[yy, xx]:
            rgb = lms_to_rgb(start * t + end * (1 - t))
            rgb = np.clip(rgb * (np.random.rand(3) * 0.2 + 0.9), 0, 1)
        else:
            new_start, new_end = lines[np.random.randint(0, len(lines))][np.random.randint(0, len(lines[0]))]
            t = np.random.rand()
            rgb = lms_to_rgb(new_start * t + new_end * (1 - t))
        r, g, b = np.clip(np.floor(rgb * 256).astype(int), 0, 255)
        return f'rgb({r}, {g}, {b})'

    draw_ishihara(drawer, output_path)
    print(text)


def draw_ishihara(colour_func: Callable, output_path: str = 'test.svg', input_path: str = 'ishihara.svg') -> None:
    """https://franciscouzo.github.io/ishihara/"""
    svg = []
    with open(input_path, 'r') as f:
        n = 0
        for line in tqdm(f):
            if line.startswith('<circle'):
                cx = float(re.search('cx="(.*?)"', line).group(1))
                cy = float(re.search('cy="(.*?)"', line).group(1))
                r = float(re.search('r="(.*?)"', line).group(1))
                svg.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{colour_func(cx / n, cy / n)}" />')
            else:
                if line.startswith('<svg'):
                    n = float(re.search('width="(.*?)"', line).group(1))
                svg.append(line)

    with open(output_path, 'w') as f:
        f.write(''.join(svg))


def generate_ishihara_set(n: int = 20, dir: str = 'ishihara') -> None:
    for i in range(n):
        char = chr(np.random.randint(65, 91)) if np.random.rand() > 0.5 else str(np.random.randint(0, 10))
        generate_ishihara(char, 'd', f'{dir}/{i}_{char}.svg')


def playground() -> None:
    h, w = 100, 200
    rgb = np.random.randint(0, 256, 3)

    mask = binary_text('Protan', h, w)
    generate_voronoi(mask, 'p', rgb, severity=0.75, n=1000)

    mask = binary_text('Deutan', h, w)
    generate_voronoi(mask, 'd', rgb, n=1000)

    mask = binary_text('Tritan', h, w)
    generate_voronoi(mask, 't', rgb, n=1000)

    h, w = 40, 80
    mask = binary_text('Abcde', h, w)
    generate_grid_inverse(mask, 'd')

    generate_ishihara(str(np.random.randint(0, 10)), 'd')

# TODO: constrain with saturation + oklab (for-real, none of this timestep BS)
