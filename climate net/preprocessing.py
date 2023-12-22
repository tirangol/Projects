import numpy as np
from scipy.ndimage import gaussian_filter

from typing import Callable, Optional
from numbers import Number

from model import device
import torch
import torch.nn as nn



resolution = (180, 360)
h, w = resolution
cells = h * w


def scale_num(x: Number) -> Number:
    """Scale a number according to the resolution."""
    return x * h // 180


def get_latitude() -> np.ndarray:
    """Return latitude."""
    # x = np.repeat(np.array(range(w)).reshape(1, w), h, axis=0)
    y = np.repeat(np.array(range(h)).reshape(h, 1), w, axis=1)
    latitude = 90 - y - scale_num(0.5)
    return latitude


latitude = get_latitude()


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def cell(start: Number, end: Number) -> np.ndarray:
    return relu(-4 * (latitude - start) * (latitude - end) / ((start - end) ** 2))


equator_n = cell(-5, 20)
hadley_n = cell(10, 40)
ferrel_n = cell(30, 60)
poles_n = cell(50, 90)

equator_s = cell(-20, 5)
hadley_s = cell(-10, -40)
ferrel_s = cell(-30, -60)
poles_s = cell(-50, -90)


def flip_matrix(matrix: np.ndarray, vertical: bool, horizontal: bool) -> np.ndarray:
    """Flip a matrix vertically or horizontally."""
    if vertical:        matrix = np.flipud(matrix)
    if horizontal:      matrix = np.fliplr(matrix)
    return matrix.copy()


def remove_na_rows(m: np.ndarray) -> np.ndarray:
    """Remove rows that contain a na value."""
    return m[~np.isnan(m).any(axis=1)]


def remove_na_cols(m: np.ndarray) -> np.ndarray:
    """Remove cols that contain a na value."""
    return m[:, ~np.any(np.isnan(m), axis=0)]


def shuffle(m: np.ndarray, direction: str, i: int = 1) -> np.ndarray:
    """Shuffle a matrix in one direction by i pixels. For instance, shifting left by 1 pixel
    moves the leftmost column to the rightmost end.

    Horizontal shuffling preserves original columns, while vertical shuffling does not.
    """
    assert direction in {"top", "bottom", "left", "right"}
    if i == 0:
        return m
    elif direction == "left":
        return np.c_[m[:, i:], m[:, :i]]
    elif direction == "right":
        return np.c_[m[:, -i:], m[:, :-i]]
    elif direction == "top":
        return np.r_[m[i:, :], m[-1, :].reshape(1, m.shape[1]).repeat(i, axis=0)]
    else:
        return np.r_[m[0, :].reshape(1, m.shape[1]).repeat(i, axis=0), m[:-i, :]]


def coordinate_angle(x: int, y: int) -> float:
    """Calculate the angle of a coordinate with respect to the origin, in degrees.
    """
    assert x != 0 or y != 0
    if x == 0:
        return 90 if y > 0 else 270
    degrees = np.arctan(y / x) * 180 / np.pi
    if x < 0:
        offset = 180
    elif y >= 0:
        offset = 0
    else:
        offset = 360
    return degrees + offset


def compare(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Compare the values in x1 and x2, which are both non-negative.
    If x2 > x1, return positive
    If x2 = x1, return 0
    If x2 < x2, return negative
    """
    return np.log((x2 + 0.1) / (x1 + 0.1))


def outer_layer(distances: np.ndarray) -> np.ndarray:
    """Given an input matrix with some "filled out" positive values and some
    "unfilled" zero values, extrapolates the outer layer of the zero values by
    averaging neighbouring positive values.
    """
    top = shuffle(distances, "top")
    bottom = shuffle(distances, "bottom")
    left = shuffle(distances, "left")
    right = shuffle(distances, "right")

    vertical = top + bottom
    vertical[np.logical_and(top != 0, bottom != 0)] /= 2

    horizontal = left + right
    horizontal[np.logical_and(left != 0, right != 0)] /= 2

    result = vertical + horizontal
    result[np.logical_and(vertical != 0, horizontal != 0)] /= 2
    return result


def outer_layer_nan(data: np.ndarray) -> None:
    data_top = shuffle(data, "top")
    data_bottom = shuffle(data, "bottom")
    data_left = shuffle(data, "left")
    data_right = shuffle(data, "right")

    mask = lambda x, y: np.logical_and(~np.isnan(x), np.isnan(y))

    top_mask = mask(data_top, data)
    bottom_mask = mask(data_bottom, data)
    left_mask = mask(data_left, data)
    right_mask = mask(data_right, data)
    all_mask = np.logical_or.reduce([top_mask, bottom_mask, left_mask, right_mask])

    counter = np.zeros(data.shape) + top_mask + bottom_mask + left_mask + right_mask

    data[all_mask] = 0
    data[top_mask] += data_top[top_mask]
    data[bottom_mask] += data_bottom[bottom_mask]
    data[left_mask] += data_left[left_mask]
    data[right_mask] += data_right[right_mask]
    data[all_mask] /= counter[all_mask]


def gaussian_filter_nan(data: np.ndarray, radius: float, land_mask: Optional[np.ndarray] = None,
                        set_water_values: float = np.nan) -> np.ndarray:
    data = data.copy()

    if land_mask is not None:
        data[land_mask == 0] = np.nan
    else:
        land_mask = ~np.isnan(data)

    while np.any(np.isnan(data)):
        outer_layer_nan(data)

    data = gaussian_filter(data, radius)
    data[land_mask == 0] = set_water_values

    return data


################################################################################
# Edges
################################################################################
def find_coastlines_straight(land: np.ndarray, hard_edge: bool = True,
                             sides: tuple[bool, bool, bool, bool] = (
                                     True, True, True, True)) -> np.ndarray:
    """Find the coastlines in a world map of 1s and 0s using L1 distance."""
    assert sum(sides) >= 1

    l, r, t, b = sides
    left = shuffle(land, "left") if l else 0
    right = shuffle(land, "right") if r else 0
    top = shuffle(land, "top") if t else 0
    bottom = shuffle(land, "bottom") if b else 0

    edges = (l + r + t + b) * land - left - right - top - bottom
    return ((edges != 0) * land).astype("int8") if hard_edge else edges


def find_coastlines_diagonal(land: np.ndarray, hard_edge: bool = True,
                             sides: tuple[bool, bool, bool, bool] = (
                                     True, True, True, True)) -> np.ndarray:
    """Find the coastlines in a world map of 1s and 0s using diagonal distance."""
    assert sum(sides) >= 1

    tl, tr, bl, br = sides
    bottom_left = shuffle(shuffle(land, "left"), "bottom") if bl else 0
    bottom_right = shuffle(shuffle(land, "right"), "bottom") if br else 0
    top_left = shuffle(shuffle(land, "left"), "top") if tl else 0
    top_right = shuffle(shuffle(land, "right"), "top") if tr else 0

    edges = (bl + br + tl + tr) * land - bottom_left - bottom_right - top_left - top_right
    return ((edges != 0) * land).astype("int8") if hard_edge else edges


def find_coastlines_square(land: np.ndarray, hard_edge: bool = True,
                           sides: tuple[bool, bool, bool, bool] = (
                                   True, True, True, True)) -> np.ndarray:
    """Find the coastlines in a world map of 1s and 0s using L∞ distance."""
    l, r, t, b = sides
    diags = (l or b, b or r, t or l, t or r)
    return np.logical_or(find_coastlines_straight(land, hard_edge, sides),
                         find_coastlines_diagonal(land, hard_edge, diags))


################################################################################
# Distances
################################################################################
def find_coastline_distances(land: np.ndarray,
                             coastline_func: Callable = find_coastlines_straight,
                             sides: tuple[bool, bool, bool, bool] = (
                                     True, True, True, True),
                             weighting: str = 'uniform') -> np.ndarray:
    """Find how many pixels away each point to a coastline in a world map of 1s
    and 0s. Incorporates both straight and square distances in an attempt to
    smooth out values."""
    assert coastline_func in {find_coastlines_straight,
                              find_coastlines_diagonal,
                              find_coastlines_square}
    assert weighting in {'uniform', 'high-biased', 'low-biased'}
    curr_land = land.copy()
    distances = np.zeros(land.shape)

    ticker = 0
    while np.any(curr_land != 0) and ticker <= w:
        edges = coastline_func(curr_land, hard_edge=True, sides=sides)
        land_outer = np.logical_and(curr_land, edges != 0)
        land_inner = np.logical_and(curr_land, edges == 0)
        curr_land[land_outer] = False
        weight = np.sqrt(ticker) * (weighting == 'low-biased') + (np.sqrt(h) - np.sqrt(ticker)) * (
                    weighting == 'high-biased')
        distances[land_inner] += 1 + weight
        ticker += 1
    return distances


def find_smooth_coastline_distances(land: np.ndarray,
                                    sides: tuple[bool, bool, bool, bool] = (
                                            True, True, True, True),
                                    weighting: str = 'uniform') -> np.ndarray:
    """Find how many pixels away each point to a coastline in a world map of 1s
    and 0s."""
    straight = find_coastline_distances(land, find_coastlines_straight, sides=sides,
                                        weighting=weighting)
    diagonal = find_coastline_distances(land, find_coastlines_diagonal, sides=sides,
                                        weighting=weighting)
    square = np.logical_or(straight, diagonal)
    return 0.5 * (straight + square)


################################################################################
# Final Functions
################################################################################
def inland_distances(land: np.ndarray,
                     coastline_func: Optional[Callable] = None,
                     edge_func: Callable = find_coastlines_straight,
                     avg_factor: int = 7,
                     sides: tuple[bool, bool, bool, bool] = (True, True, True, True),
                     weighting: str = 'uniform',
                     weighting_factor: float = 1.2) -> np.ndarray:
    """Find how many pixels away each point to a coastline in a world map of 1s
    and 0s. More sensitive to land size.

    avg_factor is the averaging/blurring factor.

    Ideal value at (360, 180) resolution is 7 on land, 10 at sea
    """
    assert weighting in {'uniform', 'low-biased', 'high-biased'}
    avg_factor = round(scale_num(avg_factor))

    assert coastline_func is None or coastline_func in {find_coastlines_straight,
                                                        find_coastlines_diagonal,
                                                        find_coastlines_square}
    assert edge_func in {find_coastlines_straight, find_coastlines_diagonal, find_coastlines_square}

    if weighting == 'low-biased':
        weights = [weighting_factor ** -i for i in range(avg_factor)]
    elif weighting == 'high-biased':
        weights = [weighting_factor ** -i for i in range(avg_factor - 1, -1, -1)]
    else:
        weights = [1 for _ in range(avg_factor)]

    curr_land = land.copy()
    curr_water = 1 - land
    coastlines = []
    for i in range(avg_factor):
        if coastline_func is None:
            coastline = find_smooth_coastline_distances(curr_land, sides=sides)
        else:
            coastline = find_coastline_distances(curr_land, coastline_func, sides=sides)
        coastlines.append(coastline * weights[i])
        edges = edge_func(curr_water)
        water_outer = np.logical_and(curr_water, edges != 0)
        curr_water[water_outer] = 0
        curr_land = 1 - curr_water

    return np.multiply(sum(coastlines) / np.sum(weights), land)


def closest_water_influence(land: np.ndarray,
                            inland_coastline_func: Optional[Callable] = None,
                            inland_edge_func: Callable = find_coastlines_straight,
                            avg_factor: int = 10, sigma: float = 2) -> np.ndarray:
    """Find how much influence the closest water body to land has."""
    sigma = round(scale_num(sigma))

    curr_land = land.copy()
    water = 1 - land
    water_influence = inland_distances(water,
                                       coastline_func=inland_coastline_func,
                                       edge_func=inland_edge_func,
                                       avg_factor=avg_factor)

    edges = find_coastlines_straight(curr_land)
    while not np.all(edges == 0):
        inner = outer_layer(water_influence) * edges
        water_influence += inner
        curr_land[inner != 0] = 0
        edges = find_coastlines_straight(curr_land)

    # Duplicate columns to account for loss of information on east/west ends of map
    # water_influence = np.c_[water_influence, water_influence[:, :6 * sigma]]
    water_influence = gaussian_filter(water_influence, sigma=sigma)
    # water_influence = np.c_[water_influence[:, w: w + 3 * sigma],
    #                         water_influence[:, 3 * sigma: w]]
    # return water_influence
    return water_influence * land


def elevation_differences(elevation: np.ndarray, radius: float) -> np.ndarray:
    """Calculate elevation differences in a circle of given radius."""
    radius = scale_num(radius)

    directions = np.zeros((8, h, w))
    # counter-clockwise, starting at right aka 0 degrees
    normalizer = np.array([0., 0., 0., 0., 0., 0., 0., 0.])

    for dist_h in {"left", "right"}:
        i = 0
        while i <= radius:
            for dist_v in {"top", "bottom"}:
                j = 0 if i != 0 else 1
                while i ** 2 + j ** 2 <= radius ** 2:
                    elevation_offset = shuffle(shuffle(elevation, dist_h, i), dist_v, j)

                    ii = i if dist_h == "right" else -i
                    jj = -j if dist_v == "top" else j
                    angle = coordinate_angle(ii, jj)

                    weights = [max(0., 1 - abs(angle / 22.5), 1 - abs((angle - 360) / 22.5))]
                    weights += [max(0., 1 - abs((angle - x) / 22.5)) for x in range(45, 360, 45)]

                    directions_offset = np.array(
                        [(elevation_offset - elevation) * x for x in weights])
                    directions += directions_offset
                    normalizer += weights
                    j += 1
            i += 1
    return directions / normalizer.reshape((8, 1, 1))


################################################################################
# Inlandness
################################################################################
# @markdown The inland measurer's blurriness
inland_scale = 7  # @param {type: "number"}

inland_st = None
inland_sq = None
inland_st_low = None
inland_sq_low = None
inland_st_high = None
inland_sq_high = None


def preprocess_inland(land: np.ndarray) -> None:
    global inland_st, inland_sq, inland_st_low, inland_sq_low, inland_st_high, inland_sq_high

    inland_st_st = inland_distances(land, find_coastlines_straight, find_coastlines_straight,
                                    avg_factor=inland_scale)
    inland_sq_st = inland_distances(land, find_coastlines_square, find_coastlines_straight,
                                    avg_factor=inland_scale)
    inland_st_sq = inland_distances(land, find_coastlines_straight, find_coastlines_square,
                                    avg_factor=inland_scale)
    inland_sq_sq = inland_distances(land, find_coastlines_square, find_coastlines_square,
                                    avg_factor=inland_scale)

    inland_st = (inland_st_st + inland_sq_st) / 2
    inland_sq = (inland_st_sq + inland_sq_sq) / 2

    inland_st_st_low = inland_distances(land, find_coastlines_straight, find_coastlines_straight,
                                        avg_factor=inland_scale, weighting='low-biased')
    inland_sq_st_low = inland_distances(land, find_coastlines_square, find_coastlines_straight,
                                        avg_factor=inland_scale, weighting='low-biased')
    inland_st_sq_low = inland_distances(land, find_coastlines_straight, find_coastlines_square,
                                        avg_factor=inland_scale, weighting='low-biased')
    inland_sq_sq_low = inland_distances(land, find_coastlines_square, find_coastlines_square,
                                        avg_factor=inland_scale, weighting='low-biased')

    inland_st_low = (inland_st_st_low + inland_sq_st_low) / 2
    inland_sq_low = (inland_st_sq_low + inland_sq_sq_low) / 2

    inland_st_st_high = inland_distances(land, find_coastlines_straight, find_coastlines_straight,
                                         avg_factor=inland_scale, weighting='high-biased')
    inland_sq_st_high = inland_distances(land, find_coastlines_square, find_coastlines_straight,
                                         avg_factor=inland_scale, weighting='high-biased')
    inland_st_sq_high = inland_distances(land, find_coastlines_straight, find_coastlines_square,
                                         avg_factor=inland_scale, weighting='high-biased')
    inland_sq_sq_high = inland_distances(land, find_coastlines_square, find_coastlines_square,
                                         avg_factor=inland_scale, weighting='high-biased')

    inland_st_high = (inland_st_st_high + inland_sq_st_high) / 2
    inland_sq_high = (inland_st_sq_high + inland_sq_sq_high) / 2


################################################################################
# Directional Inlandness
################################################################################
direction_l = (True, False, False, False)
direction_r = (False, True, False, False)
direction_t = (False, False, True, False)
direction_b = (False, False, False, True)

# @markdown The highest distance where water still affects the land
horiz_limit = 20  # @param {type: "number"}
horiz_limit = scale_num(horiz_limit)


def negative_relu(x: np.ndarray, division_factor: int = 1) -> np.ndarray:
    if division_factor == 1:
        return np.maximum(horiz_limit - x, 0) / horiz_limit
    else:
        return np.maximum(horiz_limit // division_factor - x, 0) / horiz_limit


def normalized_tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x / horiz_limit)


def blur(land: np.ndarray, x: np.ndarray) -> np.ndarray:
    return gaussian_filter_nan(x, horiz_limit / 10, land, 0)
    # return land * gaussian_filter(x, horiz_limit // 10)


l1, left = None, None
r1, right = None, None
tl1, top_left = None, None
tr1, top_right = None, None
bl1, bottom_left = None, None
br1, bottom_right = None, None
l1_r1, tl1_br1, bl1_tr1 = None, None, None
left_right, top_left_bottom_right, bottom_left_top_right = None, None, None


def preprocess_coastline(land: np.ndarray) -> None:
    global l1, left, r1, right, tl1, top_left, tr1, top_right, bl1, \
        bottom_left, br1, bottom_right, l1_r1, tl1_br1, bl1_tr1, \
        left_right, top_left_bottom_right, bottom_left_top_right

    inland_l = inland_distances(land, None, find_coastlines_straight, avg_factor=inland_scale,
                                sides=direction_l, weighting='high-biased')
    inland_r = inland_distances(land, None, find_coastlines_straight, avg_factor=inland_scale,
                                sides=direction_r, weighting='high-biased')
    inland_tl = inland_distances(land, find_coastlines_diagonal, find_coastlines_square,
                                 avg_factor=inland_scale, sides=direction_l,
                                 weighting='high-biased')
    inland_bl = inland_distances(land, find_coastlines_diagonal, find_coastlines_square,
                                 avg_factor=inland_scale, sides=direction_t,
                                 weighting='high-biased')
    inland_tr = inland_distances(land, find_coastlines_diagonal, find_coastlines_square,
                                 avg_factor=inland_scale, sides=direction_r,
                                 weighting='high-biased')
    inland_br = inland_distances(land, find_coastlines_diagonal, find_coastlines_square,
                                 avg_factor=inland_scale, sides=direction_b,
                                 weighting='high-biased')
    inland_tl_half = inland_distances(land, find_coastlines_diagonal, find_coastlines_square,
                                      avg_factor=inland_scale // 2, sides=direction_l,
                                      weighting='high-biased')
    inland_bl_half = inland_distances(land, find_coastlines_diagonal, find_coastlines_square,
                                      avg_factor=inland_scale // 2, sides=direction_t,
                                      weighting='high-biased')
    inland_tr_half = inland_distances(land, find_coastlines_diagonal, find_coastlines_square,
                                      avg_factor=inland_scale // 2, sides=direction_r,
                                      weighting='high-biased')
    inland_br_half = inland_distances(land, find_coastlines_diagonal, find_coastlines_square,
                                      avg_factor=inland_scale // 2, sides=direction_b,
                                      weighting='high-biased')

    coastline_l = find_smooth_coastline_distances(land, sides=direction_l)
    coastline_r = find_smooth_coastline_distances(land, sides=direction_r)
    coastline_tl = find_coastline_distances(land, find_coastlines_diagonal, sides=direction_l)
    coastline_bl = find_coastline_distances(land, find_coastlines_diagonal, sides=direction_r)
    coastline_tr = find_coastline_distances(land, find_coastlines_diagonal, sides=direction_t)
    coastline_br = find_coastline_distances(land, find_coastlines_diagonal, sides=direction_b)

    # West coast
    l1 = negative_relu(coastline_r, 4)
    l2 = negative_relu(inland_r)
    l3 = normalized_tanh(inland_l)
    left = (l1 + 0.2) * l3 * (l2 + 0.1)
    left = blur(land, left)

    # East coast
    r1 = negative_relu(coastline_l, 4)
    r2 = negative_relu(inland_l)
    r3 = normalized_tanh(inland_r)
    right = (r1 + 0.2) * r3 * (r2 + 0.1)
    right = blur(land, right)

    # Northwest coast
    tl1 = negative_relu(coastline_br, 4)
    tl2 = negative_relu(inland_br)
    tl3 = normalized_tanh(inland_tl_half)
    tl4 = negative_relu(inland_br_half)
    top_left = tl1 + tl2 + tl3 + tl4
    top_left[~land] = 0
    top_left = blur(land, top_left)

    # Northeast coast
    tr1 = negative_relu(coastline_bl, 4)
    tr2 = negative_relu(inland_bl)
    tr3 = normalized_tanh(inland_tr_half)
    tr4 = negative_relu(inland_bl_half)
    top_right = tr1 + tr2 + tr3 + tr4
    top_right[~land] = 0
    top_right = blur(land, top_right)

    # Southeast coast
    br1 = negative_relu(coastline_tl, 4)
    br2 = negative_relu(inland_tl)
    br3 = normalized_tanh(inland_br_half)
    br4 = negative_relu(inland_tl_half)
    bottom_right = br1 + br2 + br3 + br4
    bottom_right[~land] = 0
    bottom_right = blur(land, bottom_right)

    # Southwest coast
    bl1 = negative_relu(coastline_tr, 4)
    bl2 = negative_relu(inland_tr)
    bl3 = normalized_tanh(inland_bl_half)
    bl4 = negative_relu(inland_tr_half)
    bottom_left = bl1 + bl2 + bl3 + bl4
    bottom_left[~land] = 0
    bottom_left = blur(land, bottom_left)

    # Comparisons
    l1_r1 = compare(l1, r1)
    tl1_br1 = compare(tl1, br1)
    bl1_tr1 = compare(bl1, tr1)
    left_right = compare(left, right)
    top_left_bottom_right = compare(top_left, bottom_right)
    bottom_left_top_right = compare(top_right, bottom_left)


################################################################################
# Water Influence
################################################################################
# @markdown The water influence measurer's blurriness
water_influence_scale = 10  # @param {type: "number"}

water_influence_st = None
water_influence_sq = None


def preprocess_water_influence(land: np.ndarray) -> None:
    global water_influence_st, water_influence_sq

    water_influence_st_st = closest_water_influence(land, find_coastlines_straight,
                                                    find_coastlines_straight,
                                                    avg_factor=water_influence_scale,
                                                    sigma=water_influence_scale / 5)
    water_influence_sq_st = closest_water_influence(land, find_coastlines_square,
                                                    find_coastlines_straight,
                                                    avg_factor=water_influence_scale,
                                                    sigma=water_influence_scale / 5)

    water_influence_st_sq = closest_water_influence(land, find_coastlines_straight,
                                                    find_coastlines_square,
                                                    avg_factor=water_influence_scale,
                                                    sigma=water_influence_scale / 5)
    water_influence_sq_sq = closest_water_influence(land, find_coastlines_square,
                                                    find_coastlines_square,
                                                    avg_factor=water_influence_scale,
                                                    sigma=water_influence_scale / 5)

    water_influence_st = relu(np.log((water_influence_st_st + water_influence_sq_st) / 2 + 0.0001))
    water_influence_sq = relu(np.log((water_influence_sq_sq + water_influence_st_sq) / 2 + 0.0001))


################################################################################
# Elevation differences
################################################################################
degrees = [2, 4, 6, 8, 10, 12, 14, 16]
elevation_diffs = None


def preprocess_elevation(elevation: np.ndarray) -> None:
    global elevation_diffs
    elevation_diffs = [elevation_differences(elevation, degree) for degree in degrees]
    elevation_diffs = [[np.maximum(0, x[i]) for i in [0, 1, 3, 4, 5, 7]] for x in elevation_diffs]


################################################################################
# Final
################################################################################
inland_sq_diff = None
inland_st_high_diff = None
inland_sq_high_diff = None
inland_st_low_diff = None
inland_sq_low_diff = None
water_influence_sq_diff = None
latitude_nan = None


def final_preprocessing(land: np.ndarray) -> None:
    global inland_sq_diff, inland_st_high_diff, inland_sq_high_diff, \
        inland_st_low_diff, inland_sq_low_diff, water_influence_sq_diff, \
        latitude_nan

    inland_sq_diff = inland_sq - inland_st
    inland_st_high_diff = inland_st_high - inland_st
    inland_sq_high_diff = inland_sq_high - inland_sq - inland_st_high_diff
    inland_st_low_diff = inland_st_low - inland_st
    inland_sq_low_diff = inland_sq_low - inland_sq - inland_st_low_diff
    water_influence_sq_diff = water_influence_sq - water_influence_st
    latitude_nan = latitude.copy()
    latitude_nan[~land] = np.nan


data = None


def preprocess(elevation: np.ndarray, land: np.ndarray) -> None:
    global data

    print('Preprocessing elevation...')
    preprocess_elevation(elevation)
    print('Preprocessing water influence...')
    preprocess_water_influence(land)
    print('Preprocessing coastline...')
    preprocess_coastline(land)
    print('Preprocessing inland...')
    preprocess_inland(land)
    print('Wrapping up preprocessing...')
    final_preprocessing(land)
    print('Combining data...')
    data = np.array([elevation, latitude_nan, abs(latitude),
                     equator_n, equator_s, hadley_n, hadley_s, ferrel_n, ferrel_s, poles_n, poles_s,
                     inland_st, inland_sq_diff, inland_st_high_diff, inland_sq_high_diff,
                     inland_st_low_diff, inland_sq_low_diff,
                     left, right, bottom_left, bottom_right, top_left, top_right, l1, r1, bl1, br1,
                     tl1, tr1,
                     l1_r1, tl1_br1, bl1_tr1, left_right, top_left_bottom_right,
                     bottom_left_top_right,
                     water_influence_st, water_influence_sq_diff])
    data = np.concatenate([data, np.concatenate(elevation_diffs)])


def flatten_data(x: np.ndarray) -> np.ndarray:
    return x.reshape((x.shape[0], cells)).T.copy()


def unflatten_data(x: np.ndarray) -> np.ndarray:
    return x.T.reshape((x.shape[1], h, w)).copy()


def flip_3d(matrix: np.ndarray, v: bool, h: bool) -> np.ndarray:
    return np.array([flip_matrix(x, vertical=v, horizontal=h) for x in matrix])


def toggle_latitude_nan(nan: bool) -> None:
    global data
    if nan:
        data[1] = latitude_nan
    else:
        data[1] = latitude


features = ['elevation', 'latitude', 'absolute latitude',
            'equator n', 'equator s', 'hadley n', 'hadley s', 'ferrel n', 'ferrel s', 'poles n',
            'poles s',
            'inland l1', 'inland l∞', 'inland l1 high', 'inland l∞', 'inland l1 low',
            'inland l∞ low',
            'left coast blur', 'right coast blur', 'bottom left coast blur',
            'bottom right coast blur', 'top left coast blur', 'top right coast blur',
            'left coast', 'right coast', 'bottom left coast', 'bottom right coast',
            'top left coast', 'top right coast',
            'L-R coast', 'TL-BR coast', 'BL-TR coast', 'L-R coast blur', 'TL-BR coast blur',
            'BL-TR coast blur',
            'water influence l1', 'water influence l∞']


def get_prediction(model: nn.Module) -> np.ndarray:
    global data
    print('Predicting...')
    prediction = model(torch.tensor(flatten_data(data)).to(device))
    return prediction.detach().cpu().numpy()


def get_koppen(temp: np.ndarray, prec: np.ndarray, land: np.ndarray) -> np.ndarray:
    summer = lambda x: np.concatenate(
        [x[3:9, :h // 2, :], x[np.array([0, 1, 2, 9, 10, 11]), h // 2:, :]], axis=1)
    winter = lambda x: np.concatenate(
        [x[np.array([0, 1, 2, 9, 10, 11]), :h // 2, :], x[3:9, h // 2:, :]], axis=1)

    temp_mean = np.mean(temp, axis=0)
    temp_max = np.max(temp, axis=0)
    temp_min = np.min(temp, axis=0)

    prec_mean = np.mean(prec, axis=0)
    prec_min = np.min(prec, axis=0)
    prec_sum = np.sum(prec, axis=0)
    prec_winter_max = np.max(winter(prec), axis=0)
    prec_winter_min = np.min(winter(prec), axis=0)
    prec_summer_max = np.max(summer(prec), axis=0)
    prec_winter_sum = np.sum(winter(prec), axis=0)
    prec_summer_min = np.min(summer(prec), axis=0)
    prec_summer_sum = np.sum(summer(prec), axis=0)

    B_prec_percent = (prec_summer_sum + 0.0001) / (prec_sum + 0.0001)
    B_threshold = (20 * temp_mean + 280 * (B_prec_percent >= 0.7) + 140 * (B_prec_percent < 0.7) * (
                B_prec_percent >= 0.3))

    A_threshold = 100 - (prec_sum / 25)
    A = (temp_min >= 18) * (prec_sum > B_threshold)
    Af = A * (prec_min >= 60)
    Am = A * (prec_min < 60) * (prec_min >= A_threshold)
    As = A * (prec_summer_min < A_threshold)
    Aw = A * (prec_winter_min < A_threshold)
    del A, A_threshold

    B = (temp_max >= 10) * (prec_sum <= B_threshold)
    BS = B * (prec_sum >= 0.5 * B_threshold)
    BW = B * (prec_sum < 0.5 * B_threshold)
    B_h = temp_mean >= 18
    BSh = BS * B_h
    BSk = BS * ~B_h
    BWh = BW * B_h
    BWk = BW * ~B_h
    del B_prec_percent, B, B_h, BS, BW

    _a = temp_max >= 22
    _b = ~_a * (np.sum(temp >= 10, axis=0) >= 4)
    _c = ~_a * ~_b * (temp_min >= -38)
    _d = ~_a * ~_b * ~_c
    _w = (prec_summer_max >= 10 * prec_winter_min)
    _s = (prec_winter_max >= 3 * prec_summer_min) * (prec_summer_min < 40)
    _f = ~_w * ~_s

    C = (temp_min >= 0) * (temp_min < 18) * (temp_max >= 10) * (prec_sum > B_threshold)
    Cw = C * _w
    Cs = C * _s
    Cf = C * _f
    Cfa = Cf * _a; Cfb = Cf * _b; Cfc = Cf * _c
    Csa = Cs * _a; Csb = Cs * _b; Csc = Cs * _c
    Cwa = Cw * _a; Cwb = Cw * _b; Cwc = Cw * _c
    del C, Cw, Cs, Cf

    D = (temp_min < 0) * (temp_max >= 10) * (prec_sum > B_threshold)
    Dw = D * _w
    Ds = D * _s
    Df = D * _f
    Dfa = Df * _a; Dfb = Df * _b; Dfc = Df * _c; Dfd = Df * _d
    Dsa = Ds * _a; Dsb = Ds * _b; Dsc = Ds * _c; Dsd = Ds * _d
    Dwa = Dw * _a; Dwb = Dw * _b; Dwc = Dw * _c; Dwd = Dw * _d
    del D, Dw, Ds, Df, _w, _s, _f, _a, _b, _c, _d, B_threshold

    ET = (temp_max < 10) * (temp_max >= 0)
    EF = temp_max < 0

    k = np.empty(temp[0].shape)
    k.fill(np.nan)
    k[Af], k[Am], k[As], k[Aw] = range(0, 4)
    k[BSh], k[BSk], k[BWh], k[BWk] = range(4, 8)
    k[Cfa], k[Cfb], k[Cfc], k[Csa], k[Csb], k[Csc], k[Cwa], k[Cwb], k[Cwc] = range(8, 17)
    k[Dfa], k[Dfb], k[Dfc], k[Dfd] = range(17, 21)
    k[Dsa], k[Dsb], k[Dsc], k[Dsd] = range(21, 25)
    k[Dwa], k[Dwb], k[Dwc], k[Dwd] = range(25, 29)
    k[EF], k[ET] = range(29, 31)
    k[land == 0] = np.nan
    return k


def get_trewartha(temp: np.ndarray, prec: np.ndarray, elevation: np.ndarray,
                  land: np.ndarray) -> np.ndarray:
    summer = lambda x: np.concatenate(
        [x[3:9, :h // 2, :], x[np.array([0, 1, 2, 9, 10, 11]), h // 2:, :]], axis=1)
    winter = lambda x: np.concatenate(
        [x[np.array([0, 1, 2, 9, 10, 11]), :h // 2, :], x[3:9, h // 2:, :]], axis=1)

    temp_mean = np.mean(temp, axis=0)
    temp_max = np.max(temp, axis=0)
    temp_min = np.min(temp, axis=0)

    prec_min = np.min(prec, axis=0)
    prec_sum = np.sum(prec, axis=0)
    prec_winter_min = np.min(winter(prec), axis=0)
    prec_summer_max = np.max(summer(prec), axis=0)
    prec_summer_sum = np.sum(summer(prec), axis=0)

    temp_gt_10 = np.sum(temp >= 10, axis=0)
    prec_threshold = 10 * (temp_mean - 10) + 300 * (prec_summer_sum + 0.0001) / (prec_sum + 0.0001)

    A = (temp_min >= 18) * (prec_sum >= 2 * prec_threshold)
    Ar = A * (np.sum(prec >= 60, axis=0) > 10)
    Am = A * ~Ar * (prec_min < 60) * (prec_min >= (2500 - prec_sum) / 25)
    Aw = A * ~Ar * ~Am * (np.sum(winter(prec) < 60, axis=0) > 2)
    As = A * ~Ar * ~Aw * ~Aw

    B = (prec_sum < 2 * prec_threshold) * (temp_gt_10 >= 3)
    BS = B * (prec_sum >= prec_threshold)
    BSh = BS * (temp_gt_10 >= 8)
    BSk = BS * (temp_gt_10 < 8)
    BW = B * ~BS
    BWh = BW * (temp_gt_10 >= 8)
    BWk = BW * (temp_gt_10 < 8)
    del BS, BW

    _a = temp_max >= 22
    _b = ~_a
    _w = (prec_sum < 890) * (prec_winter_min < 30) * (prec_winter_min < prec_summer_max / 3)
    _s = (prec_sum < 890) * (prec_winter_min < 30) * (prec_winter_min < prec_summer_max / 3)
    _f = ~_s * ~_w  # (prec_min >= 30)

    C = (temp_gt_10 >= 8) * (prec_sum >= 2 * prec_threshold) * (temp_min < 18)
    Cf = C * _f
    Cw = C * _w
    Cs = C * _s
    Cfa = Cf * _a; Cfb = Cf * _b
    Cwa = Cw * _a; Cwb = Cw * _b
    Csa = Cs * _a; Csb = Cs * _b
    del Cf, Cw, Cs

    D = (temp_gt_10 < 8) * (temp_gt_10 >= 4) * (prec_sum >= 2 * prec_threshold)
    DC = D * (temp_min < 0)
    DO = D * ~DC
    DCfa = DC * _f * _a; DCfb = DC * _f * _b
    DCsa = DC * _s * _a; DCsb = DC * _s * _b
    DCwa = DC * _w * _a; DCwb = DC * _w * _b
    DOfa = DO * _f * _a; DOfb = DO * _f * _b
    DOsa = DO * _s * _a; DOsb = DO * _s * _b
    DOwa = DO * _w * _a; DOwb = DO * _w * _b
    del DC, DO

    E = (temp_gt_10 < 4) * (temp_gt_10 >= 1)
    EC = E * (temp_min < -10)
    EO = E * (temp_min >= -10)
    F = temp_max < 10
    Ft = F * (temp_max > 0)
    Fi = F * (temp_max < 0)
    del _a, _b, _w, _s, _f

    new_temp = temp + 0.0056 * elevation * (cell(-65, 65) ** 0.5)
    new_temp_max = np.max(new_temp, axis=0)
    new_temp_min = np.min(new_temp, axis=0)
    new_temp_gt_10 = np.sum(new_temp >= 10, axis=0)
    new_A = (new_temp_min >= 18) * (prec_sum >= 2 * prec_threshold)
    new_B = (prec_sum < 2 * prec_threshold) * (new_temp_gt_10 >= 3)
    new_C = (new_temp_gt_10 >= 8) * (prec_sum >= 2 * prec_threshold) * (new_temp_min < 18)
    new_D = (new_temp_gt_10 < 8) * (new_temp_gt_10 >= 4) * (prec_sum >= 2 * prec_threshold)
    new_E = (new_temp_gt_10 < 4) * (new_temp_gt_10 >= 1)
    new_F = new_temp_max < 10
    del new_temp, new_temp_max, new_temp_min, new_temp_gt_10

    k = np.empty((h, w))
    k.fill(np.nan)
    k[Ar], k[Am], k[Aw], k[As] = range(4)
    k[BSh], k[BSk], k[BWh], k[BWk] = range(4, 8)
    k[Cfa], k[Cfb], k[Cwa], k[Cwb], k[Csa], k[Csb] = range(8, 14)
    k[DCfa], k[DCfb], k[DCsa], k[DCsb], k[DCwa], k[DCwb] = range(14, 20)
    k[DOfa], k[DOfb], k[DOsa], k[DOsb], k[DOwa], k[DOwb] = range(20, 26)
    k[EC], k[EO] = 26, 27
    k[Ft], k[Fi] = 28, 29
    k[np.logical_or.reduce([A != new_A, B != new_B, C != new_C,
                            D != new_D, E != new_E, F != new_F]) * (elevation >= 2500)] = 30
    k[land == 0] = np.nan
    del A, B, C, D, E, F, new_A, new_B, new_C, new_D, new_E, new_F
    return k
