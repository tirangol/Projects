from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt


def generate_noisy_grid(mask: NDArray, cvd_type: str, texture_gradient: bool = True, resolution: int = 10) -> NDArray:
    """
    Generate a noisy grid where each pixel in mask is amplified by resolution.
    The content in each pixel follows a confusion line (TRUE) or random noise (FALSE) based on the mask value,
    so that the difference is only apparent for a cvd_type
    """
    h, w = mask.shape
    data = np.empty((h * resolution, w * resolution, 3))
    
    gradient = get_gradient(cvd_type)

    if texture_gradient:
        t = np.add.outer(np.arange(resolution), np.arange(resolution), dtype=float)[:, :, None]
        t /= np.max(t)

    for i in tqdm(range(h * w)):
        x, y = divmod(i, h)
        ind = int(np.clip(np.round(x / w * len(gradient)), 0, len(gradient) - 1))
        colour = np.clip(gradient[ind] * (np.random.rand(3) * 0.2 + 0.9), 0, 1)

        if mask[y, x]:
            to_rgb = lms_to_rgb
            start, end = confusion_line(colour, cvd_type)
        else:
            to_rgb = lambda x: x ** (1 / 2.4)
            if cvd_type == 'p':
                direction = np.array([np.random.rand() * 0.2 + 0.2, np.random.rand() * 0.8 - 0.7, 0])
            elif cvd_type == 'd':
                direction = np.array([np.random.rand() * 0.4 + 0.1, np.random.rand() * 0.7, 0])
            else:
                raise Exception("Method not supported for tritans")
            
            start, end = confusion_line(colour, direction)
            start, end = lms_to_rgb(start) ** 2.4, lms_to_rgb(end) ** 2.4

        if np.random.rand() > 0.5:
            start, end = end, start

        if not texture_gradient:
            t = np.random.rand(resolution, resolution, 1)

        square = to_rgb(start * (1 - t) + end * t)
        y0 = y * resolution
        y1 = (y + 1) * resolution
        x0 = x * resolution
        x1 = (x + 1) * resolution
        data[y0:y1, x0:x1, :] = square

    return data


def playground() -> None:
    h, w = 40, 80
    mask = binary_text('Abcde', h, w)
    grid = generate_noisy_grid(mask, 'd')
    grid = generate_noisy_grid(mask, 'd', False)
    plt.imshow(grid)