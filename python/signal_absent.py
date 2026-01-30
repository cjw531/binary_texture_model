import numpy as np
import matplotlib.pyplot as plt
import cv2


def bin_1_f(muimg, sigimg, numimgs=1):
    """
    Samples a background noise with a 1/f noise characteristic.

    Parameters
    ----------
    muimg : np.ndarray
        Mean image (same shape as the desired output image).
    sigimg : np.ndarray
        Standard deviation image (same shape as muimg).
    numimgs : int, optional
        Number of images to generate.

    Returns
    -------
    img : np.ndarray
        If numimgs == 1: a 2D image with same shape as muimg.
        If numimgs > 1: a 2D array with shape (muimg.size, numimgs),
        where each column is a flattened image.
    """

    # Get dimensions of the image (same as MATLAB's dim = size(muimg))
    dim = muimg.shape  # (rows, cols)

    # Generate frequency grid
    # In MATLAB: [x, y] = meshgrid(-(dim(2)/2):(dim(2)/2-1), -(dim(1)/2):(dim(1)/2-1));
    cols = dim[1]
    rows = dim[0]
    x = np.arange(-cols / 2, cols / 2) # length = cols
    y = np.arange(-rows / 2, rows / 2) # length = rows
    x, y = np.meshgrid(x, y) # x, y each have shape (rows, cols)

    radius = np.sqrt(x**2 + y**2) + 1 # Radius in frequency domain (add 1 to avoid division by zero)
    scaling_factor = 2.5 # hardcoded scaling factor
    power_spectrum = 1.0 / (radius ** scaling_factor) # 1/f^scaling_factor power spectrum

    if numimgs == 1:
        n = np.random.randn(*dim) + 1j * np.random.randn(*dim) # Complex white noise: real and imaginary parts ~ N(0, 1)

        # Apply Fourier transform:
        # MATLAB: img_freq = fftshift(fft2(n)) .* power_spectrum;
        # Python: np.fft.fft2, then np.fft.fftshift to move zero-freq to center
        img_freq = np.fft.fftshift(np.fft.fft2(n)) * power_spectrum

        # Inverse Fourier transform:
        # MATLAB: ifft2(ifftshift(img_freq))
        # Python: np.fft.ifftshift, then np.fft.ifft2
        img = np.fft.ifft2(np.fft.ifftshift(img_freq))

        img = np.real(img) # Take real part (imag part is numerical noise)
        img = img * sigimg + muimg # Scale by sigimg and shift by muimg (element-wise)

    else:
        # For multiple images, we store flattened images as columns.
        img = np.zeros((muimg.size, numimgs), dtype=np.float64) # MATLAB: img = zeros(prod(dim), numimgs);

        for i in range(numimgs):
            n = np.random.randn(*dim) + 1j * np.random.randn(*dim) # Complex white noise

            img_freq = np.fft.fftshift(np.fft.fft2(n)) * power_spectrum # Fourier transform, apply power spectrum
            im = np.fft.ifft2(np.fft.ifftshift(img_freq)) # Inverse Fourier transform

            im = np.real(im) # Real part
            im = im * sigimg + muimg # Apply local mean and std
            img[:, i] = im.ravel() # Flatten into a column and store

    # Convert to double (np.float64) and normalize to [0, 1]
    img = img.astype(np.float64)
    img_min = img.min()
    img_max = img.max()
    img = (img - img_min) / (img_max - img_min)

    # Compute global mean and std over all elements
    t = img.mean()
    sigma_t = img.std() * 0.1

    # Apply sigmoid: img = 1 ./ (1 + exp(-(img - t) / sigma_t));
    img = 1.0 / (1.0 + np.exp(-(img - t) / sigma_t))

    # Apply final scaling and offset:
    # delta_mu = 0.35; mu_a = 0.3; img = img * delta_mu + mu_a;
    delta_mu = 0.35
    mu_a = 0.3
    img = img * delta_mu + mu_a

    return img


if __name__ == "__main__":
    """ DEBUG / TESTING PURPOSES ONLY """

    N = 64 # image dimension

    # Create inputs matching the MATLAB call:
    # muimg = zeros(64,64), sigimg = ones(64,64), numimgs = 1
    muimg = np.zeros((N, N), dtype=np.float64)
    sigimg = np.ones((N, N), dtype=np.float64)
    numimgs = 1

    # Call the function (equivalent to n = Bin_1_f(zeros(64,64),ones(64,64),1);)
    n = bin_1_f(muimg, sigimg, numimgs)

    # Convert to 8-bit grayscale for saving
    n_8bit = (n * 255).astype(np.uint8)

    # Save with OpenCV (this is equivalent to imwrite/imsave)
    cv2.imwrite("signal_absent.png", n_8bit)

    bin_1_f_noise = n + 0.1 * np.random.randn(*n.shape)

    bin_1_f_noise_8bit = (bin_1_f_noise * 255).astype(np.uint8)
    cv2.imwrite("signal_absent_noise.png", bin_1_f_noise_8bit)
