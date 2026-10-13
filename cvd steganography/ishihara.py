from utils import *
import re
from tqdm import tqdm
from typing import Callable


def draw_ishihara(colour_func: Callable[[float, float], str], output_path: str = 'test.svg', input_path: str = 'ishihara.svg') -> None:
    """
    Create an Ishihara plate based on an SVG template file from https://franciscouzo.github.io/ishihara/

    colour_func takes in a (x, y) coordinate [0-1] and outputs a string colour
    """
    svg = []
    with open(input_path, 'r') as f:
        n = 0
        for line in tqdm(f):
            if line.startswith('<circle'):
                cx = float(re.search('cx="(.*?)"', line).group(1))
                cy = float(re.search('cy="(.*?)"', line).group(1))
                r = float(re.search('r="(.*?)"', line).group(1))
                svg.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{colour_func(cx / n, cy / n)}" />\n')
            else:
                if line.startswith('<svg'):
                    n = float(re.search('width="(.*?)"', line).group(1))
                svg.append(line)

    with open(output_path, 'w') as f:
        f.write(''.join(svg))


def distractor_lines(rgb: tuple[NDArray, NDArray], n: int = 5) -> list[tuple[Callable[[NDArray], NDArray], NDArray, NDArray]]:
    """
    Generate "distractor" lines for a confusion line which CVD people see as distinct

    rgb (2, 3): a start and end RGB value [0-1] 
    n: number of lines to generate
    """
    lines = []
    start_rgb, end_rgb = rgb
    start, end = rgb_to_oklab(start_rgb), rgb_to_oklab(end_rgb)
    
    unit = np.random.normal(size=(2, n, 3))
    unit /= np.linalg.norm(unit, axis=-1, keepdims=True)
    sign = 2 * (np.random.rand(n, 3) > 0.5) - 1
    
    for i in range(n):
        # Jitter start/end RGB values in OKLAB space
        # Then draw a line between them in RGB, linear-RGB, and LMS space (randomly chosen)
        r = np.random.rand()
        if r > 2 / 3:
            min_radius, radius = 0.07, 0.09
        else:
            min_radius, radius = 0.04, 0.06

        perturb = unit[0, i] * (radius - min_radius) + min_radius
        p1 = oklab_to_rgb(np.clip(start + sign[i] * perturb, 0, 1))

        perturb = unit[1, i] * (radius - min_radius) + min_radius
        p2 = oklab_to_rgb(np.clip(end + sign[i] * perturb, 0, 1))

        if r > 2 / 3:
            lines.append((lms_to_rgb, rgb_to_lms(p1), rgb_to_lms(p2)))
        elif r > 1 / 3:
            lines.append((lambda x: x ** (1 / 2.4), p1 ** 2.4, p2 ** 2.4))
        else:
            lines.append((lambda x: x, p1, p2))
    return lines


def simulate_cvd_ishihara(input_path: str, output_path: str, type: str, degree: float, method: str) -> None:
    """Open an Ishihara SVG, simulate a specific CVD type on it, and save the result as another file"""
    svg = []
    with open(input_path, 'r') as f:
        for line in tqdm(f):
            if line.startswith('<circle'):
                cx = float(re.search('cx="(.*?)"', line).group(1))
                cy = float(re.search('cy="(.*?)"', line).group(1))
                r = float(re.search('r="(.*?)"', line).group(1))
                fill = str(re.search('fill="rgb\\((.*?)\\)"', line).group(1))

                rgb = np.array([float(x) for x in fill.split(', ')]) / 255
                new_rgb = simulate_cvd(rgb[None, None], type, degree, method)[0, 0] * 255
                new_rgb = f'rgb({new_rgb[0]}, {new_rgb[1]}, {new_rgb[2]})'

                svg.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{new_rgb}" />\n')
            else:
                svg.append(line)

    with open(output_path, 'w') as f:
        f.write(''.join(svg))


def generate_ishihara(mask: NDArray, cvd_type: str, output_path: str = 'test.svg') -> None:
    """Generate a CVD-visible Ishihara plate based on a mask"""
    _, start, end = useful_confusion_line(cvd_type)
    h, w = mask.shape

    lines = distractor_lines((lms_to_rgb(start), lms_to_rgb(end)), 10)

    def drawer(x: float, y: float) -> str:
        xx = np.clip(int(x * w), 0, w - 1)
        yy = np.clip(int(y * h), 0, h - 1)

        if mask[yy, xx]:
            t = (np.random.beta(4, 4) + 0.5) % 1
            rgb = lms_to_rgb(start * t + end * (1 - t))
            rgb = np.clip(rgb * (np.random.rand(3) * 0.1 + 0.95), 0, 1)
        else:
            to_rgb, new_start, new_end = lines[np.random.randint(0, len(lines))]
            t = (np.random.beta(4, 4) + 0.5) % 1
            rgb = to_rgb(new_start * t + new_end * (1 - t))
        r, g, b = np.clip(np.floor(rgb * 256).astype(int), 0, 255)
        return f'rgb({r}, {g}, {b})'

    draw_ishihara(drawer, output_path)


def generate_ishihara_set(cvd_type: str, n: int = 20, dir: str = 'ishihara', method: str = 'vienot') -> None:
    """Generate a test set of Ishihara images + CVD simulations"""
    for i in range(n):
        char = chr(np.random.randint(65, 91))  # upper-case letters
        # char = str(np.random.randint(0, 10))  # digits

        path = f'{dir}/{i}_{char}.svg'
        path_sim = f'{dir}/{i}_{char}_sim.jpg'
        mask = binary_text(char, 500, 500)
        generate_ishihara(mask, cvd_type, path)
        simulate_cvd_ishihara(path, path_sim, cvd_type, 1, method)


def generate_regular_ishihara(mask: NDArray, cvd_type: str, severity: float = 0.5, output_path: str = 'test.svg') -> None:
    """Generate a CVD-invisible Ishihara plate based on a mask"""
    h, w = mask.shape
    _, start, end = useful_confusion_line(cvd_type, 0.3)
    t0 = 0.5 - severity / 2
    t1 = 0.5 + severity / 2
    start, end = lms_to_rgb(start * t0 + end * (1 - t0)), lms_to_rgb(start * t1 + end * (1 - t1))

    def drawer(x: float, y: float) -> str:
        xx = np.clip(int(x * w), 0, w - 1)
        yy = np.clip(int(y * h), 0, h - 1)

        colour = start if mask[yy, xx] else end
        rgb = np.clip(colour * (np.random.rand(3) * 0.4 + 0.8), 0, 1)
        r, g, b = np.clip(np.floor(rgb * 256).astype(int), 0, 255)
        return f'rgb({r}, {g}, {b})'

    draw_ishihara(drawer, output_path)


def playground() -> None:
    size = 500
    mask = binary_text('A', size, size)
    generate_ishihara(mask, 'd')
    generate_regular_ishihara(mask, 'd')
    generate_ishihara_set('d')