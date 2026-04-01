import os
import argparse
import numpy as np
import cv2

from tqdm import tqdm
from signal_absent import bin_1_f
from signal_present import disk_signal

np.random.seed(0)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate simulated medical image datasets."
    )

    # Output base folder (relative or absolute). Default is current directory.
    parser.add_argument(
        "outdir",
        default="./data/",
        help="Base output folder where datasets will be saved (relative or absolute). Default: current directory",
    )

    # Required positional arguments (user must enter them)
    parser.add_argument(
        "num_datasets",
        type=int,
        help="Number of datasets to be generated (required)",
    )
    parser.add_argument(
        "image_dim",
        type=int,
        help="Image dimension (e.g., 64 for 64x64) (required)",
    )

    # Define which dataset(s) to generate
    parser.add_argument(
        "--absent",
        action="store_true",
        help="Generate signal absent images",
    )

    parser.add_argument(
        "--present",
        action="store_true",
        help="Generate signal present images",
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.07,
        help="Alpha value (float, default: 0.07)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help="Radius value (float, default: 2.0)",
    )

    return parser.parse_args()

def save_config(args):
    """
    Save the command-line arguments to <outdir>/config.txt.
    Each line is written as: key = value
    """
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)  # ensure directory exists

    config_path = os.path.join(outdir, "config.txt")
    with open(config_path, "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key} = {value}\n")

def run(data_path: str, num_datasets: int, image_dim: int, alpha: float, radius: float, generate_absent: bool, generate_present: bool) -> None:
    print("========================================")
    print(f"Generating {num_datasets} datasets in {data_path}")
    print(f"Image dimension: {image_dim}")
    print(f"Alpha: {alpha}")
    print(f"Radius: {radius}")
    print(f"Generate signal absent images: {generate_absent}")
    print(f"Generate signal present images: {generate_present}")
    print("========================================")

    # create subfolders depending on requested datasets
    os.makedirs(data_path, exist_ok=True)
    if generate_absent:
        absent_clean_png_dir = os.path.join(data_path, "signal_absent_clean_png")
        os.makedirs(absent_clean_png_dir, exist_ok=True)
        absent_clean_npy_dir = os.path.join(data_path, "signal_absent_clean_npy")
        os.makedirs(absent_clean_npy_dir, exist_ok=True)
        absent_noise_png_dir = os.path.join(data_path, "signal_absent_noise_png")
        os.makedirs(absent_noise_png_dir, exist_ok=True)
        absent_noise_npy_dir = os.path.join(data_path, "signal_absent_noise_npy")
        os.makedirs(absent_noise_npy_dir, exist_ok=True)
    if generate_present:
        present_clean_png_dir = os.path.join(data_path, "signal_present_clean_png")
        os.makedirs(present_clean_png_dir, exist_ok=True)
        present_clean_npy_dir = os.path.join(data_path, "signal_present_clean_npy")
        os.makedirs(present_clean_npy_dir, exist_ok=True)
        present_noise_png_dir = os.path.join(data_path, "signal_present_noise_png")
        os.makedirs(present_noise_png_dir, exist_ok=True)
        present_noise_npy_dir = os.path.join(data_path, "signal_present_noise_npy")
        os.makedirs(present_noise_npy_dir, exist_ok=True)

    # base input arrays
    muimg = np.zeros((image_dim, image_dim), dtype=np.float64)
    sigimg = np.ones((image_dim, image_dim), dtype=np.float64)

    # pre-generate the signal (assuming the signal is identical across the dataset)
    if generate_present:
        signal = disk_signal(image_dim, alpha, radius) # generate signal present image

    ''' generate datasets '''
    for i in tqdm(range(num_datasets), total=num_datasets, desc="Generating datasets"):
        img_absent = bin_1_f(muimg, sigimg, numimgs=1) # generate signal absent image, base background

        ''' Signal Absent Image Generation '''
        if generate_absent:
            img_absent_noise = img_absent + 0.1 * np.random.randn(*img_absent.shape) # add Gaussian noise

            # save signal absent in numpy
            np.save(os.path.join(absent_clean_npy_dir, f"lumpy_{i}.npy"), img_absent)
            np.save(os.path.join(absent_noise_npy_dir, f"lumpy_noise_{i}.npy"), img_absent_noise)

            # save signal absent image in 8-bit img
            img_absent_8bit = (img_absent * 255).astype(np.uint8) # convert to 8-bit grayscale for saving
            cv2.imwrite(os.path.join(absent_clean_png_dir, f"lumpy_{i}.png"), img_absent_8bit) # save image
            img_absent_noise_8bit = (img_absent_noise * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(absent_noise_png_dir, f"lumpy_noise_{i}.png"), img_absent_noise_8bit)
        
        ''' Signal Present Image Generation '''
        if generate_present:
            img_present = img_absent + signal
            img_present_noise = img_present + 0.1 * np.random.randn(*img_present.shape)

            # save signal present in numpy
            np.save(os.path.join(present_clean_npy_dir, f"lumpy_{i}.npy"), img_present)
            np.save(os.path.join(present_noise_npy_dir, f"lumpy_noise_{i}.npy"), img_present_noise)

            # save signal present image in 8-bit img
            img_present_8bit = (img_present * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(present_clean_png_dir, f"lumpy_{i}.png"), img_present_8bit)
            img_present_noise_8bit = (img_present_noise * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(present_noise_png_dir, f"lumpy_noise_{i}.png"), img_present_noise_8bit)


if __name__ == "__main__":
    """
    Usage:
        python ./python/generate_dataset.py <outdir> <num_datasets> <image_dim> [--alpha ALPHA] [--radius RADIUS] [--absent] [--present]
    
    Example:
        python ./python/generate_dataset.py ./data/ 20000 64 --alpha 0.07 --radius 2 --absent --present
    """

    args = parse_args()
    save_config(args)
    run(data_path=args.outdir, num_datasets=args.num_datasets, image_dim=args.image_dim, alpha=args.alpha, radius=args.radius, \
        generate_absent=args.absent, generate_present=args.present)
    