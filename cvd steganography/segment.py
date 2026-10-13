from utils import *
import matplotlib.pyplot as plt


# https://en.wikipedia.org/wiki/Segment_display#Characters
CHAR_TO_7SEGMENT = {
    '0': 'abcdef',
    '1': 'bc',
    '2': 'abged',
    '3': 'abgcd',
    '4': 'fgbc',
    '5': 'afgcd',
    '6': 'afgecd',
    '7': 'abc',
    '8': 'abcdefg',
    '9': 'abcdfg',
    'A': 'afbgec',
    'B': 'cdefg',
    'C': 'afed',
    'D': 'bcdeg',
    'E': 'afged',
    'F': 'afge',
    'G': 'afedc',
    'H': 'fegbc',
    'I': 'bc',
    'J': 'bcd',
    'L': 'fed',
    'N': 'egc',
    'O': 'abcdef',
    'P': 'afbge',
    'R': 'eg',
    'S': 'afgcd',
    'U': 'fedbc',
}


# https://en.wikipedia.org/wiki/Fourteen-segment_display
# a, b, c, d, e, f, g1, g2, h, i, j, k, l, m
# a, b, c, d, e, f, g, h, i, j, k, l, m, n
CHAR_TO_14SEGMENT = {
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

def draw_7segment(get_colour: Callable[[int, bool], NDArray], text: str) -> None:
    text = text.upper()
    assert all(x in CHAR_TO_7SEGMENT for x in text), f"Text has forbidden characters. Allowed: {''.join(list(CHAR_TO_7SEGMENT.keys()))}"

    text_h, text_w = 200, 140
    char_h, char_w = text_h + 20, text_w + 20

    def draw_line(x: list[int], y: list[int], colour: NDArray) -> None:
        plt.plot(x, y, color=colour, linewidth=7.5, solid_capstyle='round')

    def draw_segment(segment: str, offset: tuple[int, int], colour: NDArray) -> None:
        y, x = offset
        if segment == 'a':
            draw_line([x + 20, x + text_w - 20],            [y + text_h - 10, y + text_h - 10], colour)
        elif segment == 'b':
            draw_line([x + text_w - 10, x + text_w - 10],   [y + text_h // 2 + 15, y + text_h - 25], colour)
        elif segment == 'c':
            draw_line([x + text_w - 10, x + text_w - 10],   [y + 25, y + text_h // 2 - 15], colour)
        elif segment == 'd':
            draw_line([x + 20, x + text_w - 20],            [y + 10, y + 10], colour)
        elif segment == 'e':
            draw_line([x + 10, x + 10],                     [y + 25, y + text_h // 2 - 15], colour)
        elif segment == 'f':
            draw_line([x + 10, x + 10],                     [y + text_h // 2 + 15, y + text_h - 25], colour)
        elif segment == 'g':
            draw_line([x + 20, x + text_w - 20],            [y + text_h // 2, y + text_h // 2], colour)
        else:
            raise ValueError(segment)

    segments = [chr(x) for x in range(ord('a'), ord('g') + 1)]
    chars_per_line = 12

    len_x = min(chars_per_line, len(text))
    len_y = np.ceil(len(text) / chars_per_line)

    plt.figure(figsize=(1.4 * len_x, 2 * len_y))
    for i, x in enumerate(text):
        y_offset, x_offset = divmod(i, chars_per_line)
        active_segments = set(char for char in CHAR_TO_7SEGMENT[x])
        for char in segments:
            colour = get_colour(i, char in active_segments)
            draw_segment(char, (y_offset * -char_h, x_offset * char_w), colour)

    plt.xlim([0, len_x * char_w])
    plt.ylim([-(len_y - 1) * char_h, char_h])
    plt.axis(False)


def draw_14segment(get_colour: Callable[[int, bool], NDArray], text: str) -> None:
    text = text.upper()
    assert all(x in CHAR_TO_14SEGMENT for x in text), f"Text has forbidden characters. Allowed: {''.join(list(CHAR_TO_14SEGMENT.keys()))}"

    text_h, text_w = 200, 140
    char_h, char_w = text_h + 20, text_w + 20

    def draw_line(x: list[int], y: list[int], colour: NDArray) -> None:
        plt.plot(x, y, color=colour, linewidth=7.5, solid_capstyle='round')

    def draw_segment(segment: str, offset: tuple[int, int], colour: NDArray) -> None:
        y, x = offset
        if segment == 'a':
            draw_line([x + 20, x + text_w - 20],            [y + text_h - 10, y + text_h - 10], colour)
        elif segment == 'b':
            draw_line([x + text_w - 10, x + text_w - 10],   [y + text_h // 2 + 15, y + text_h - 25], colour)
        elif segment == 'c':
            draw_line([x + text_w - 10, x + text_w - 10],   [y + 25, y + text_h // 2 - 15], colour)
        elif segment == 'd':
            draw_line([x + 20, x + text_w - 20],            [y + 10, y + 10], colour)
        elif segment == 'e':
            draw_line([x + 10, x + 10],                     [y + 25, y + text_h // 2 - 15], colour)
        elif segment == 'f':
            draw_line([x + 10, x + 10],                     [y + text_h // 2 + 15, y + text_h - 25], colour)
        elif segment == 'g':
            draw_line([x + 20, x + text_w // 2 - 10],       [y + text_h // 2, y + text_h // 2], colour)
        elif segment == 'h':
            draw_line([x + text_w // 2 + 10, x + text_w - 20],  [y + text_h // 2, y + text_h // 2], colour)
        elif segment == 'i':
            draw_line([x + 25, x + text_w // 2 - 15],           [y + text_h - 25, y + text_h // 2 + 15], colour)
        elif segment == 'j':
            draw_line([x + text_w // 2, x + text_w // 2],       [y + text_h // 2 + 15, y + text_h - 25], colour)
        elif segment == 'k':
            draw_line([x + text_w // 2 + 15, x + text_w - 25],  [y + text_h // 2 + 15, y + text_h - 25], colour)
        elif segment == 'l':
            draw_line([x + 25, x + text_w // 2 - 15],           [y + 25, y + text_h // 2 - 15], colour)
        elif segment == 'm':
            draw_line([x + text_w // 2, x + text_w // 2],       [y + 25, y + text_h // 2 - 15], colour)
        elif segment == 'n':
            draw_line([x + text_w // 2 + 15, x + text_w - 25],  [y + text_h // 2 - 15, y + 25], colour)
        else:
            raise ValueError(segment)

    segments = [chr(x) for x in range(ord('a'), ord('n') + 1)]
    chars_per_line = 12

    len_x = min(chars_per_line, len(text))
    len_y = np.ceil(len(text) / chars_per_line)

    plt.figure(figsize=(1.4 * len_x, 2 * len_y))
    for i, x in enumerate(text):
        y_offset, x_offset = divmod(i, chars_per_line)
        active_segments = set(char for char in CHAR_TO_14SEGMENT[x])
        for char in segments:
            colour = get_colour(i, char in active_segments)
            draw_segment(char, (y_offset * -char_h, x_offset * char_w), colour)

    plt.xlim([0, len_x * char_w])
    plt.ylim([-(len_y - 1) * char_h, char_h])
    plt.axis(False)


def sanity_check_7segment() -> None:
    def drawer(id: int, active: bool) -> NDArray:
        if active:
            return np.array([1, 0, 0])
        return np.array([0.9, 0.9, 0.9])
    draw_7segment(drawer, ''.join(list(CHAR_TO_7SEGMENT.keys())))


def sanity_check_14segment() -> None:
    def drawer(id: int, active: bool) -> NDArray:
        if active:
            return np.array([1, 0, 0])
        return np.array([0.9, 0.9, 0.9])
    draw_14segment(drawer, ''.join(list(CHAR_TO_14SEGMENT.keys())))


def generate_segment(text: str, cvd_type: str, use14: bool = False) -> None:
    """Generate some segment displays where the content is CVD-visible"""
    curr = None
    start, end = None, None
    start_rgb, end_rgb = None, None
    
    def drawer(id: int, active: bool) -> NDArray:
        nonlocal curr, start, end, start_rgb, end_rgb

        if curr != id:
            _, start, end = useful_confusion_line(cvd_type, 0.3)
            start_rgb, end_rgb = lms_to_rgb(start), lms_to_rgb(end)
            curr = id

        if active:
            t = (np.random.beta(4, 4) + 0.5) % 1
            return lms_to_rgb(start * (1 - t) + end * t)

        c0 = rgb_to_oklab(start_rgb if np.random.rand() > 0.5 else end_rgb)
        c0_sim = simulate_cvd(c0[None, None, :], cvd_type)[0, 0, :]

        c = np.random.rand(3)
        c_sim = simulate_cvd(c[None, None, :], cvd_type)[0, 0, :]
        dist = np.sum((rgb_to_oklab(c) - c0) ** 2) ** 0.5
        c_sim_dist = np.sum((rgb_to_oklab(c_sim) - c0_sim) ** 2) ** 0.5
        while dist < 0.2 or dist > 0.4 or c_sim_dist < 0.3:
            c = np.random.rand(3)
            c_sim = simulate_cvd(c[None, None, :], cvd_type)[0, 0, :]
            dist = np.sum((rgb_to_oklab(c) - c0) ** 2) ** 0.5
            c_sim_dist = np.sum((rgb_to_oklab(c_sim) - c0_sim) ** 2) ** 0.5
        return c

    if use14:
        draw_14segment(drawer, text)
    else:
        draw_7segment(drawer, text)


def playground() -> None:
    generate_segment('thequickfoxjumpsoverthelazydog', 'd', True)
    generate_segment('0987654321', 'p')
