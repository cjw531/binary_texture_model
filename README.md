# binary_texture_model
Generates simulated medical images for (1) signal present; and (2) signal absent, with and without noise

<hr>

## Python
Refer to: [`./python/`](./python) folder

### Generate Signal Absent Dataset
[`./python/signal_absent.py`](./python/signal_absent.py)

### Generate Signal Present Dataset
[`./python/signal_present.py`](./python/signal_present.py)

### Runner / Wrapper
[`./python/generate_dataset.py`](./python/generate_dataset.py)

#### Usage: 
```
python ./python/generate_dataset.py <outdir> <num_datasets> <image_dim> [--alpha ALPHA] [--radius RADIUS] [--absent] [--present]
```

#### Example:
```
python ./python/generate_dataset.py ./data/ 20000 64 --alpha 0.07 --radius 2 --absent --present
```

<hr>

## MATLAB
Refer to: [`./matlab/`](./matlab) folder

### Generate Signal Absent Dataset
[`./matlab/Bin_1_f.m`](./matlab/Bin_1_f.m)

### Generate Signal Present Dataset
[`./matlab/DiskSignal.m`](./matlab/DiskSignal.m)

### Runner / Wrapper
[`./matlab/bin1fdisk_gen.m`](./matlab/bin1fdisk_gen.m)
- For mass data generation
- Also can add noise to the generated images
