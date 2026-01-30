import numpy as np
import matplotlib.pyplot as plt
import cv2

from signal_absent import bin_1_f


def disk_signal(dim, alpha, radius, pos=None):
    """
    Generate an image of dimensions dim with a white (value=alpha) disk.

    Parameters
    ----------
    dim : int or tuple/list of two ints
        Image dimensions. If a single int is given, image is square (dim x dim).
    alpha : float
        Intensity value inside the disk.
    radius : float
        Radius of the disk (in pixels).
    pos : tuple/list of two floats (x, y), optional
        Position of the disk center in (x, y) coordinates.
        If None, the disk is centered in the image (like the MATLAB code).

    Returns
    -------
    img : np.ndarray
        2D image array of shape (dim[0], dim[1]) with a disk of value alpha.
    """

    # If dim is a single int, make it square as [dim, dim]
    if isinstance(dim, int):
        dim = (dim, dim)
    elif len(dim) == 1:
        dim = (dim[0], dim[0])

    # If pos is not given, center the disk in the image
    # MATLAB: pos = fliplr((dim - [1 1]) / 2);
    # Here, dim is (rows, cols). We want (x, y), so we flip.
    if pos is None:
        # Convert to numpy array to do arithmetic, then flip [rows, cols] -> [cols, rows]
        dim_arr = np.array(dim, dtype=float)
        pos = np.flip((dim_arr - 1) / 2.0)  # gives [x_center, y_center]
        # pos[0] is x, pos[1] is y
    else:
        pos = np.array(pos, dtype=float)

    # Create coordinate grid:
    # MATLAB:
    # x = 0:(dim(2)-1);
    # y = 0:(dim(1)-1);
    # [X,Y] = meshgrid(x,y);
    rows, cols = dim
    x = np.arange(cols)     # 0 .. cols-1
    y = np.arange(rows)     # 0 .. rows-1
    X, Y = np.meshgrid(x, y)  # X,Y both shape (rows, cols)

    # Initialize the image
    img = np.zeros(dim, dtype=float)

    # Create a circular mask with value 1 inside the radius:
    # circle = sqrt((X-pos(1)).^2 + (Y-pos(2)).^2) <= radius;
    # Recall: pos = (x_center, y_center)
    circle = np.sqrt((X - pos[0])**2 + (Y - pos[1])**2) <= radius

    # Set pixels inside the circle to alpha
    img[circle] = alpha

    return img


if __name__ == "__main__":
    """ DEBUG / TESTING PURPOSES ONLY """

    N = 64 # image dimension

    # Create inputs matching the MATLAB call:
    # muimg = zeros(64,64), sigimg = ones(64,64), numimgs = 1
    muimg = np.zeros((N, N), dtype=np.float64)
    sigimg = np.ones((N, N), dtype=np.float64)
    numimgs = 1

    n = bin_1_f(muimg, sigimg, numimgs)
    sig = disk_signal(N, 0.07, 2) # disk signal function call: sig = DiskSignal(64,0.07,2);
    sys = n + sig

    # Convert to 8-bit grayscale for saving
    sys_8bit = (sys * 255).astype(np.uint8)

    # Save with OpenCV (this is equivalent to imwrite/imsave)
    cv2.imwrite("signal_present.png", sys_8bit)

    disk_signal_noise = sys + 0.1 * np.random.randn(*sys.shape)

    disk_signal_noise_8bit = (disk_signal_noise * 255).astype(np.uint8)
    cv2.imwrite("signal_present_noise.png", disk_signal_noise_8bit)

