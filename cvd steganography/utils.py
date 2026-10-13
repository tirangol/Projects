import numpy as np
from numpy.typing import NDArray
import Geometry3D as geo3d
import plotly.graph_objects as go
from typing import Callable, Iterable
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import Voronoi
from matplotlib.patches import Polygon


####################################################################################################
# Colour Spaces
####################################################################################################
# Smith & Pokorny 1975, LMS to XYZ
# Judd-Vos Linear sRGB to XYZ
LINEAR_SRGB_TO_LMS = np.array([
    [17.8824, 43.5161, 4.11935],
    [3.45565, 27.1554, 3.86714],
    [0.0299566, 0.184309, 1.46709]
]).T
LMS_TO_LINEAR_SRGB = np.array([
    [8.09443559e-02, -1.30504315e-01, 1.16720585e-01],
    [-1.02485056e-02, 5.40193131e-02, -1.13614715e-01],
    [-3.65297605e-04, -4.12162784e-03, 6.93513240e-01]
]).T


def rgb_to_lms(rgb: NDArray) -> NDArray:
    """Convert SRGB (..., 3), matrix of floats [0-1], to LMS"""
    return (rgb ** 2.4) @ LINEAR_SRGB_TO_LMS


def lms_to_rgb(lms: NDArray) -> NDArray:
    """Convert LMS (..., 3), matrix of floats, to SRGB [0-1]"""
    return np.clip(lms @ LMS_TO_LINEAR_SRGB, 0, 1) ** (1 / 2.4)


# https://en.wikipedia.org/wiki/Oklab_color_space
OKLAB_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005]
])
OKLAB_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660]
])


def rgb_to_oklab(rgb: NDArray) -> NDArray:
    """
    rgb is (..., 3) of RGB values from [0, 1]
    return is (..., 3) of OKLAB values
    """
    lms = (rgb ** 2.4) @ OKLAB_M1.T
    return np.cbrt(lms) @ OKLAB_M2.T


def oklab_to_rgb(oklab: NDArray) -> NDArray:
    """
    oklab is (..., 3) of OKLAB values
    return is (..., 3) of RGB values from [0, 1]
    """
    lms = np.linalg.solve(OKLAB_M2, oklab.reshape(-1, 3).T).T ** 3
    rgb = np.linalg.solve(OKLAB_M1, lms.T).T
    return (np.clip(rgb, 0, 1) ** (1 / 2.4)).reshape(oklab.shape)


####################################################################################################
# CVD Simulation
####################################################################################################
def cvd_vienot_1999(x: NDArray, type: str, degree: float = 1.0) -> NDArray:
    if type == 'p':
        m = np.array([[1 - degree, 2.02344377 * degree, -2.52580405 * degree],
                      [0, 1, 0],
                      [0, 0, 1]]).T
    elif type == 'd':
        m = np.array([[1, 0, 0],
                      [0.49420696 * degree, 1 - degree, 1.24826995 * degree],
                      [0, 0, 1]]).T
    elif type == 't':
        m = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [-0.01224491 * degree, 0.07203435 * degree, 1 - degree]]).T
    else:
        raise ValueError()
    return x @ m


def cvd_brettel_1997(x: NDArray, type: str, degree: float = 1.0) -> NDArray:
    if type == 'p':
        h1 = np.array([
            [1 - degree, 2.18595384 * degree, -4.10029338 * degree],
            [0., 1., 0.],
            [0., 0., 1.]
        ])
        h2 = np.array([
            [1 - degree, 2.27376148 * degree, -5.92721645 * degree],
            [0., 1., 0.],
            [0., 0., 1.]
        ])
        n_sep_plane = np.array([0., 0.012864, -0.267648])
    elif type == 'd':
        h1 = np.array([
            [1., 0., 0.],
            [0.4574662 * degree, 1 - degree, 1.87574564 * degree],
            [0., 0., 1.]
        ])
        h2 = np.array([
            [1., 0., 0.],
            [0.43979987 * degree, 1 - degree, 2.60678902 * degree],
            [0., 0., 1.]
        ])
        n_sep_plane = np.array([-0.012864, 0., 0.53232])
    elif type == 't':
        h1 = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [-0.00254865 * degree, 0.0531321 * degree, 1 - degree]
        ])
        h2 = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [-0.05574292 * degree, 0.15892917 * degree, 1 - degree]
        ])
        n_sep_plane = np.array([0.267648, -0.53232, 0.])
    else:
        raise ValueError()

    img_h1 = x @ h1.T
    img_h2 = x @ h2.T
    h2_indices = np.dot(x, n_sep_plane) < 0
    img_h1[h2_indices] = img_h2[h2_indices]
    return img_h1


def simulate_cvd(x: NDArray, type: str, degree: float = 1, method: str = 'vienot') -> NDArray:
    """
    https://daltonlens.org/understanding-cvd-simulation/
    https://github.com/DaltonLens/DaltonLens-Python/tree/master
    https://daltonlens.org/opensource-cvd-simulation/
    https://daltonlens.org/colorblindness-simulator

    x:      (H, W, 3), an RGB image [0, 1]
    type:   'd', 'p', or 't'
    degree: [0, 1]
    output: (H, W, 3), an RGB image simulating CVD
    """
    assert x.ndim == 3 and x.shape[2] == 3
    if method == 'vienot':
        f = cvd_vienot_1999
    elif method == 'brettel':
        f = cvd_brettel_1997
    else:
        raise ValueError(method)

    lms = rgb_to_lms(x)
    sim = f(lms, type, degree)
    rgb = lms_to_rgb(sim)
    return rgb


####################################################################################################
# Lines
####################################################################################################
def confusion_line(rgb: NDArray, type: str | NDArray) -> NDArray:
    """
    rgb:    (3,), [0, 1]
    type:   'p' (protan), 'd' (deutan), 't' (tritan), or a custom direction vector (3,)
    output: (2, 3), start/end points of confusion line. Two LMS values.
    """
    if not isinstance(type, str):
        e = type
    elif type == 'p':
        e = np.array([1, 0, 0])
    elif type == 'd':
        e = np.array([0, 1, 0])
    elif type == 't':
        e = np.array([0, 0, 1])
    else:
        e = type

    krgb = rgb_to_lms(np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        rgb
    ]))
    k, r, g, b, lms = krgb
    shape = geo3d.Parallelepiped(geo3d.Point(*k),
                                 geo3d.Vector(*(r - k)),
                                 geo3d.Vector(*(g - k)),
                                 geo3d.Vector(*(b - k)))
    line = geo3d.Line(geo3d.Point(*lms), geo3d.Point(*(lms + e)))
    segment = geo3d.intersection(shape, line)
    return np.array([[p.x, p.y, p.z] for p in segment])


def useful_confusion_line(cvd_type: str, threshold: float = 0.2) -> tuple[NDArray, NDArray, NDArray]:
    """Generate a confusion line in LMS space that is meaningfully long"""
    
    def sample_confusion_line(cvd_type: str) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        rgb = np.random.rand(3)
        start, end = confusion_line(rgb, cvd_type)
        dist = np.sum((rgb_to_oklab(lms_to_rgb(start)) - rgb_to_oklab(lms_to_rgb(end))) ** 2) ** 0.5
        return rgb, start, end, dist

    rgb, start, end, dist = sample_confusion_line(cvd_type)
    while dist < threshold:
        rgb, start, end, dist = sample_confusion_line(cvd_type)

    if np.random.rand() > 0.5:
        start, end = end, start
    return rgb, start, end


def get_gradient(cvd_type: str, n: int = 50) -> list[NDArray]:
    """Return a size-n list of (3,) RGB points representing a gradient in OKLAB space"""
    
    def sample_colour(compare_to: NDArray | None = None) -> Iterable[NDArray]:
        c = np.random.rand(3)
        start, end = confusion_line(c, cvd_type)
        dist = np.sum((rgb_to_oklab(lms_to_rgb(start)) - rgb_to_oklab(lms_to_rgb(end))) ** 2) ** 0.5
        if compare_to is not None:
            c_dist = np.sum((rgb_to_oklab(compare_to) - rgb_to_oklab(c)) ** 2) ** 0.5
            return c, dist, c_dist
        return c, dist

    c0, dist = sample_colour()
    while dist < 0.2:
        c0, dist = sample_colour()

    c1, dist, c_dist = sample_colour(c0)
    while dist < 0.2 or c_dist < 0.4:
        c1, dist, c_dist = sample_colour(c0)

    o0 = rgb_to_oklab(c0)
    o1 = rgb_to_oklab(c1)
    return [oklab_to_rgb(o0 * (1 - t / n) + o1 * t / n) for t in range(n + 1)]


def plot_lines(original: tuple[Callable[[NDArray], NDArray], NDArray, NDArray],
               lines: list[tuple[Callable[[NDArray], NDArray], NDArray, NDArray]],
               linear_rgb: bool = True, names: list[str] | None = None, n: int = 20) -> None:
    """
    original:   (2, 3) of start/end values in some colour space
    lines:      (2, 3)[] distractor start/end values in some colour space
    linear_rgb: plot in linear_rgb space instead of rgb space
    names:      optional list of names for each distractor line
    """
    fig = go.Figure()
    exp = 2.4 if linear_rgb else 1    
    sample_points = lambda to_rgb, start, end: np.stack([to_rgb(start * (1 - i / n) + end * (i / n)) ** exp for i in range(n + 1)], axis=0)

    # Plot original line
    to_rgb, start, end = original
    orig = sample_points(to_rgb, start, end)
    fig.add_trace(go.Scatter3d(
        x = orig[:, 0],
        y = orig[:, 1],
        z = orig[:, 2],
        mode = "lines",
        line = {'color': "rgb(0, 0, 0)", 'width': 5},
        name = 'original'
    ))

    # Plot distractor lines
    rgbs_all = [orig]
    for idx, (to_rgb, start, end) in enumerate(lines):
        rgbs = sample_points(to_rgb, start, end)
        rgbs_all.append(rgbs)
        fig.add_trace(go.Scatter3d(
            x = rgbs[:, 0],
            y = rgbs[:, 1],
            z = rgbs[:, 2],
            mode = "lines",
            line = {'color': "rgba(0, 0, 0, 0.2)", 'width': 5},
            name = None if names is None else names[idx],
        ))

    # Plot coloured points
    rgbs_all = np.concatenate(rgbs_all, axis=0)
    fig.add_trace(go.Scatter3d(
        x = rgbs_all[:, 0],
        y = rgbs_all[:, 1],
        z = rgbs_all[:, 2],
        mode = "markers",
        marker = {'size': 5, 'color': [f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})" for r, g, b in rgbs_all ** (1 / exp)]},
        name = 'points'
    ))

    suffix = ' (linear)' if linear_rgb else ''
    fig.update_layout(scene = {
        'xaxis_title': f"Red{suffix}",
        'yaxis_title': f"Green{suffix}",
        'zaxis_title': f"Blue{suffix}",
        'xaxis': {'range': [0, 1]},
        'yaxis': {'range': [0, 1]},
        'zaxis': {'range': [0, 1]},
        'aspectmode': 'cube'
    })

    fig.show()


####################################################################################################
# Helper
####################################################################################################
def binary_text(text: str, h: int, w: int, font_path: str | None = None) -> NDArray:
    """Helper function, generate binary mask of a given size with text"""
    image = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(image)

    def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        if font_path is None:
            return ImageFont.load_default(size=size)
        return ImageFont.truetype(font_path, size=size)

    # Find biggest font size that doesn't exceed 80% canvas width/height
    size = 1
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, load_font(size))
    while x1 - x0 < w * 0.8 and y1 - y0 < h * 0.8:
        size *= 2
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, load_font(size))
    while x1 - x0 > w * 0.8 or y1 - y0 > h * 0.8:
        size -= 1
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, load_font(size))

    draw.text((w // 2, h // 2), text, fill=1, font=load_font(size), spacing=0, anchor='mm', align='center')
    return np.array(image) > 0


def generate_voronoi(mask: NDArray, cvd_type: str, rgb: NDArray,
                     severity: float = 0.5, draw_outline: bool = False, n: int = 5000) -> None:
    """Using the mask, generate a voronoi polygon scene that is invisible for a given cvd_type"""
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
            y, x = np.round(np.mean(polygon, axis=0)).astype(int)
            
            if mask[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)]:
                color = np.clip(c1 * rand[i], 0, 1)
            else:
                color = np.clip(c2 * rand[i], 0, 1)
            
            if draw_outline:
                ax.fill(*zip(*polygon), color=color, edgecolor='white', linewidth=1)

            polygon[:, 0] = h - polygon[:, 0]
            ax.add_patch(Polygon(np.flip(polygon, axis=1), closed=True, facecolor=color))

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)