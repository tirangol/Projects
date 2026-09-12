import numpy as np
from numpy.typing import NDArray
import Geometry3D as geo3d

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

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


def rgb_to_oklab(rgb: NDArray) -> NDArray:
    """
    srgb is (3, ...) of RGB values from [0, 1]
    return is (3, ...) of OKLAB values
    """
    r, g, b = rgb ** 2.4
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    l = np.cbrt(l)
    m = np.cbrt(m)
    s = np.cbrt(s)

    L = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s
    a = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s
    b = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s

    return np.stack([L, a, b])


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

    x:      (H, W, 3), an RGB image
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


def confusion_line(rgb: NDArray, type: str) -> NDArray:
    """
    rgb:    (3,), [0, 1]
    type:   'p', 'd', or 't'
    output: (2, 3), start/end points of confusion line. Two LMS values.
    """
    if type == 'p':
        e = np.array([1, 0, 0])
    elif type == 'd':
        e = np.array([0, 1, 0])
    elif type == 't':
        e = np.array([0, 0, 1])
    else:
        raise ValueError()

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


def confusion_line_custom(rgb: NDArray, e: NDArray) -> NDArray:
    """
    rgb:    (3,), [0, 1], must be linear-rgb (i.e. rgb ** 2.4)
    output: (2, 3), start/end points of confusion line. Two LMS values.
    """
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
