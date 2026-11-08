# Binary Texture Model

This repository generates synthetic medical-image-like textures for binary detection experiments:

- **Signal absent:** a stochastic lumpy background.
- **Signal present:** the same type of background with a circular disk signal.
- **Clean and noisy variants:** optional additive Gaussian measurement noise.

The repository includes equivalent Python and MATLAB implementations.

## Mathematical Background Generation Model

Let an image have dimensions $H \times W$. The generator first creates complex white noise:

$$
z(x,y) = z_\mathrm{r}(x,y) + i z_\mathrm{i}(x,y),
\qquad
z_\mathrm{r}, z_\mathrm{i} \sim \mathcal{N}(0,1),
$$

with independent samples. Let $(u,v)$ denote centered spatial-frequency coordinates and define:

$$
\rho(u,v) = \sqrt{u^2+v^2}+1.
$$

The `+1` prevents a singularity at the zero-frequency bin. A radial frequency weighting is then applied:

$$
A(u,v) = \rho(u,v)^{-\beta},
\qquad \beta = 2.5.
$$

For each realization, the Fourier-domain texture is:

$$
\widehat{b}(u,v)
  = A(u,v)\,
    \mathcal{F}\{z\}(u,v),
$$

and the real-valued background texture is obtained with the inverse transform:

$$
b_0(x,y)
  = \mathrm{Re}\left[
      \mathcal{F}^{-1}\{\widehat{b}\}(x,y)
    \right].
$$

Because high spatial frequencies are attenuated by $A(u,v)$, the result has large-scale correlated structure rather than pixel-independent noise. Strictly speaking, the implementation applies a $1/\rho^\beta$ **amplitude** filter; the corresponding power spectral density scales approximately as $1/\rho^{2\beta}$.

The generated texture can be modulated by spatial mean and standard-deviation images. For mean image $\mu(x,y)$ and standard-deviation image $\sigma(x,y)$:

$$
b_1(x,y) = \sigma(x,y)b_0(x,y)+\mu(x,y).
$$

The default dataset uses $\mu(x,y)=0$ and $\sigma(x,y)=1$.

### Normalization & Contrast Mapping

The background is min-max normalized:

$$
\widetilde{b}(x,y)
  = \frac{b_1(x,y)-\min(b_1)}
          {\max(b_1)-\min(b_1)}.
$$

It is then passed through a sigmoid centered at the global mean $t=\mathrm{mean}(\widetilde{b})$:

$$
s(x,y)
  = \frac{1}
    {1+\exp\left(-\frac{\widetilde{b}(x,y)-t}{\sigma_t}\right)},
\qquad
\sigma_t = 0.1\,\mathrm{std}(\widetilde{b}).
$$

Finally, the contrast and offset are set by $\Delta\mu=0.35$ and $\mu_a=0.30$:

$$
b(x,y) = \Delta\mu\,s(x,y)+\mu_a.
$$

Thus, apart from numerical variation, the background is mapped into the intensity interval $[0.30, 0.65]$.

## Signal-Present Model

The signal is a disk with amplitude $\alpha$, radius $r$, and center $(x_0,y_0)$. For pixel coordinates $(x,y)$, define:

$$
d(x,y)=\sqrt{(x-x_0)^2+(y-y_0)^2}.
$$

The disk image is:

$$
c(x,y)=
\begin{cases}
\alpha, & d(x,y)\le r,\\
0,      & d(x,y)>r.
\end{cases}
$$

By default, the disk is centered in the image. A signal-present image is the additive composition:

$$
I_\mathrm{present}(x,y)=b(x,y)+c(x,y),
$$

while a signal-absent image is:

$$
I_\mathrm{absent}(x,y)=b(x,y).
$$

## Measurement Noise Model

For each clean image $I$, the noisy counterpart is generated using Gaussian distribution:

$$
I_\mathrm{noisy}(x,y)
  = I(x,y)+\varepsilon(x,y),
\qquad
\varepsilon(x,y)\sim\mathcal{N}(0,\,0.1^2),
$$

independently for each pixel.

## Repository layout

```text
python/
├── environment.yml
├── generate_dataset.py   # Dataset runner
├── signal_absent.py      # 1/f-style background generator
└── signal_present.py     # Disk signal and signal-present generator
matlab/
├── Bin_1_f.m             # 1/f-style background generator
├── DiskSignal.m          # Disk signal generator
└── bin1fdisk_gen.m       # MATLAB batch-generation example
```

---

## Python

- Refer to: [`./python/`](./python) folder.

## Conda Environment Setup

Install dependencies [`./python/environment.yml`](./python/environment.yml):

```bash
conda env create -f python/environment.yml
conda activate binary_texture_model
```

## Generate Dataset

Usage:
```bash
python ./python/generate_dataset.py <outdir> <num_datasets> <image_dim> [--alpha ALPHA] [--radius RADIUS] [--absent] [--present]
```

Example usage:
```bash
python python/generate_dataset.py ./data 20000 64 \
  --alpha 0.07 --radius 2 --absent --present
```

The generator writes a `config.txt` file and, for each requested class, creates directories for clean/noisy PNG and NumPy files:

```text
OUTDIR/
├── config.txt
├── signal_absent_clean_png/
├── signal_absent_clean_npy/
├── signal_absent_noise_png/
├── signal_absent_noise_npy/
├── signal_present_clean_png/
├── signal_present_clean_npy/
├── signal_present_noise_png/
└── signal_present_noise_npy/
```

Use `--absent`, `--present`, or both. The default signal amplitude is `alpha = 0.07`, and the default disk signal radius is `2` pixels.

---

## MATLAB

- Refer to: [`./matlab/`](./matlab) folder.
- [`./matlab/Bin_1_f.m`](./matlab/Bin_1_f.m) generates the correlated background.
- [`./matlab/DiskSignal.m`](./matlab/DiskSignal.m) generates the disk signal.
- [`./matlab/bin1fdisk_gen.m`](./matlab/bin1fdisk_gen.m) demonstrates batch generation.

The MATLAB batch script can be adapted to change the number of images, image dimensions, output paths, signal amplitude, and signal radius.

The MATLAB implementation provides the same core components.

---

# Notes

## Reproducibility

- The Python runner initializes NumPy's random generator with `np.random.seed(0)`.
- The MATLAB batch example uses `rng(0)`. Keep these settings unchanged when reproducing the example datasets.

## Misc.

- PNG files are saved as 8-bit grayscale images after multiplying intensities by 255.
- NumPy files preserve the generated floating-point arrays.
- The scripts create output directories automatically.
- Existing output files with the same names may be overwritten.