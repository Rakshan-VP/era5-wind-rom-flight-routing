# Non-Intrusive Reduced-Order Modeling of Downward-Camera Image Sequences for Vision-Based Quadrotor Pose Estimation

Course project for AS 5401 (Reduced-Order Modeling).

## Problem Statement

A quadrotor's downward-facing camera produces a new image every frame. Each 64×64 grayscale image is a vector of 4096 pixels, but the drone's motion has only four degrees of freedom here (x, y, z, yaw), so the data are highly redundant. This project builds reduced-order models (ROMs) of the image sequence using Proper Orthogonal Decomposition (POD) for the low-dimensional basis, and compares two non-intrusive methods for the dynamics of the reduced variables:

- **DMD** (Dynamic Mode Decomposition): `z(k+1) = A z(k)`
- **Operator Inference**: `z(k+1) = A z(k) + H (z(k) ⊗ z(k)) + B u(k)`

**Question:** how many POD modes are needed to recover the drone's position and yaw, and which method predicts future frames more accurately and stably on a held-out part of the flight?

### Inputs and Outputs

| | Description | Size |
|---|---|---|
| **Input (full-order state)** | Downward-camera frame, grayscale, resized to 64×64, mean-subtracted, flattened | 4096 |
| **Reduced state** | POD coefficients `z(k) = Uᵣᵀ x(k)` | r (to be varied) |
| **Input (optional)** | Drone control input `u(k)`, used only in Operator Inference | to be decided |
| **Output 1** | Predicted future frames, reconstructed as `x̂ = Uᵣ ẑ` | 4096 |
| **Output 2** | Estimated pose (x, y, z, yaw), from a regression `p ≈ C z` fitted on training data | 4 |

The ground-truth pose is **not** used to build the ROMs. It is used only to fit the pose regression and to evaluate the results.

### Evaluation Plan

- **Split:** first 70% of the frames for training, last 30% as a held-out interval.
- **Metrics:** relative image reconstruction and prediction error vs. r and prediction horizon, pose RMSE vs. r, long-horizon stability, and computational cost.
- **Limitation:** train and test frames come from the same flight, so the test is easier than evaluation on a new flight, and the pose estimate is tied to this scene.

## Data

**Dataset:** [Mid-Air](https://midair.ulg.ac.be/), a synthetic multi-modal dataset for low-altitude drone flights.

| Item | Choice |
|---|---|
| Environment | Kite |
| Trajectory | 0000 |
| Climate | Sunny |
| Sensor stream | Down RGB (downward-facing camera) |
| Ground truth | Pose from the accompanying hdf5 file |
| Download size | about 620 MB |

The data are **not included in this repository**. To get them:

1. Go to the [Mid-Air download page](https://midair.ulg.ac.be/download.html).
2. Under Image Type, select only **Down RGB**. Under Kite environment, select trajectory **0000** with **sunny** only.
3. Submit your email and save the link file you receive as `download_config.txt`.
4. From inside the `data/` folder, run:

```bash
wget --content-disposition -x -nH -i path_to/download_config.txt
find . -name "*.zip" | while read filename; do unzip -o -d $(dirname "$filename") "$filename"; done;
```

### Data License and Citation

The Mid-Air dataset is released under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/). It is used here for non-commercial coursework and is not redistributed. If you cite or build on it, please cite:

> M. Fonder and M. Van Droogenbroeck, "Mid-Air: A Multi-Modal Dataset for Extremely Low Altitude Drone Flights," *IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2019.

## Folder Structure

```
uav-camera-pose-rom/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── data/                  # Mid-Air files (git-ignored, see Data section)
├── src/
│   ├── datarep/           # data representation: load frames and hdf5, build snapshot matrix
│   ├── maincode/          # POD, DMD, Operator Inference, pose regression
│   └── postprocessing/    # metrics and plotting scripts
└── results/
    └── figures/           # plots used in the report
```

## Pipeline

1. **`datarep`:** load frames, convert to grayscale, resize to 64×64, subtract the mean, and assemble the snapshot matrix `X` (4096 × number of frames). Load the ground-truth pose.
2. **`maincode`:** compute the SVD (POD), fit DMD and Operator Inference on the training coefficients, and fit the pose regression.
3. **`postprocessing`:** singular value decay, POD modes, reconstruction and prediction errors, pose RMSE, and ground-truth comparison plots.

## Requirements

Python 3.9 or later with NumPy, SciPy, h5py, Pillow (or OpenCV), and Matplotlib.

```bash
pip install -r requirements.txt
```

## License

Code in this repository is released under the MIT License (see `LICENSE`). The Mid-Air dataset is covered by its own license, described above.