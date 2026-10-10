# Single-Camera Calibration and Sphere AR — Assignment 3

This project calibrates a Basler camera from chessboard photographs, evaluates the influence of acquisition choices, exports camera poses and the chessboard to PLY, and projects a virtual sphere onto photographs and live camera frames.

## Assignment status

| Requirement | Implementation | Validation |
| --- | --- | --- |
| Intrinsic/extrinsic calibration | `calibrate.py` | 19 photographs successfully calibrated |
| Effects of image count, board coverage, and view angle | `experiments.py` | 25 calibration runs with a shared held-out set |
| Camera centers and chessboard in PLY | `export_ply.py` | Exported point clouds |
| Projection of interesting 3D points | `project_sphere.py` | Shaded sphere rendered on 19 photographs; combined triangle mesh exported |
| Real-time demonstration | `live_sphere.py` | Offline pose estimation and rendering passed on 19 photographs; live hardware performance still requires verification |

## Repository layout

```text
assignment3/                         Original 19 BMP photographs
camera_calibration/
  calibrate.py                       Calibration and reprojection evaluation
  experiments.py                     Reproducible subset experiments
  export_ply.py                      Board and camera pose visualization
  project_sphere.py                  Static sphere AR and combined PLY mesh
  live_sphere.py                     Basler/pypylon real-time sphere AR
  results/
    calibration.json                 Human-readable calibration parameters
    calibration.npz                  NumPy parameter archive
    per_image_errors.csv             Per-image calibration residuals
    camera_centers.csv               Camera centers and corresponding filenames
    calibration_scene.ply            Board, camera centers, and frusta
    calibration_scene_with_sphere.ply Combined mesh, if exported to results/
    camera_centers.ply               Camera center points
    chessboard.ply                   Board point cloud
    corners/                        Corner detection visualizations
    undistorted_example.jpg          Lens distortion correction example
    experiments/                    Run records, subset membership, and summaries
    ar_sphere/                      Static AR outputs when using commands below
    live_demo/                      Offline checks, snapshots, and recordings
```

Output subdirectories are created as needed. Their locations depend on `--output`; not every directory must exist before running the corresponding command.

## Setup

Use Python with NumPy and OpenCV. Basler acquisition additionally requires pypylon and the applicable pylon camera drivers.

```powershell
pip install -r requirements.txt
```

Run the following commands from the repository root (`lab3`).

## Calibration

```powershell
python camera_calibration/calibrate.py --images assignment3 --output camera_calibration/results
```

The board contains **10 columns × 7 rows of inner corners**, corresponding to 11 × 8 squares. The images are 2592 × 1944 pixels. Intrinsics and distortion are shared across views, while each photograph has its own rotation and translation.

The current full-dataset reprojection RMSE is **2.175707 pixels**:

```text
K = [[3692.588493,    0.000000, 1269.994118],
     [   0.000000, 3689.660601,  988.861738],
     [   0.000000,    0.000000,    1.000000]]

Distortion [k1, k2, p1, p2, k3]:
[-0.541522, 0.342266, 0.004883, -0.003261, -0.957076]
```

The square edge length is fixed at **1**. All translations, camera centers, sphere dimensions, and exported PLY coordinates use **one square edge as the unit of length**. No physical size measurement or millimeter conversion is required. The sphere radius is 0.5 units. Keep square display size, camera optics, and sensor mode consistent across captures.

Coordinates satisfy `X_camera = R @ X_board + t`. The first detected inner corner is the board origin; X follows columns and Y follows rows. Camera centers are `C = -R.T @ t`. Rodrigues rotation vectors use radians, not Euler angles. A symmetric chessboard may reverse detected corner ordering; exported poses should not be interpreted as a globally tracked acquisition trajectory.

## Controlled calibration experiments

```powershell
python camera_calibration/experiments.py --images assignment3 --calibration camera_calibration/results/calibration.json --output camera_calibration/results/experiments
```

### Protocol

- All 19 images use the same corner detection and subpixel refinement settings and the same five-coefficient distortion model.
- Four fixed images are held out from every experimental intrinsic calibration: `14-35-51`, `14-38-53`, `14-52-44`, and `14-56-42` (filename time suffixes).
- The remaining 15 images form the training pool. On each held-out image, intrinsics and distortion are fixed and only its pose is estimated using `solvePnP`.
- Validation RMSE is the square root of the mean squared 2D Euclidean corner residual across all held-out corners. It measures image-space reprojection consistency, not physical measurement accuracy.
- Image-count experiments use 5, 10, and 15 training images. The 5- and 10-image settings each use 10 random subsets (seed 20261010); the 15-image setting uses the entire pool once.
- Coverage experiments compare six images whose normalized board centroid is closest to the image center against six whose centroid is farthest from it. This is a **position proxy**, not a direct measurement of full-frame corner coverage.
- Angle experiments compare six lowest-tilt images against six highest-tilt images. Tilt is derived from the previously computed full-dataset pose: `acos(abs(R[2,2]))`. It is an approximate grouping label, not an independently measured ground-truth angle.
- Every run's training/validation filenames and estimated parameters are retained in `experiments.json` and `runs.csv`.

### Results

| Factor | Group | Training images | Runs | Training RMSE mean (px) | Validation RMSE mean (px) | Validation RMSE SD (px) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Image count | 5 images | 5 | 10 | 2.501 | 1.524 | 0.159 |
| Image count | 10 images | 10 | 10 | 2.295 | 1.360 | 0.038 |
| Image count | 15 images | 15 | 1 | 2.355 | 1.325 | N/A |
| Board position | Central | 6 | 1 | 3.002 | 1.498 | N/A |
| Board position | Peripheral | 6 | 1 | 1.378 | 1.387 | N/A |
| View angle | Low tilt | 6 | 1 | 2.621 | 3.288 | N/A |
| View angle | High tilt | 6 | 1 | 2.466 | 1.488 | N/A |

SD describes variation across sampled subsets, not uncertainty across independent capture sessions. A single run has no estimated repetition variability; machine-readable summaries encode its SD as zero.

### Discussion

**Image count.** Increasing the training set from 5 to 10 images reduced average held-out RMSE by approximately 10.8%; using all 15 training images reduced it by approximately 13.1% relative to the five-image average. The subset-to-subset standard deviation of estimated `fx` fell from 76.22 to 26.25 pixels between five and ten images. More images improved stability in this dataset, but training RMSE did not decrease monotonically: adding images also adds different observations and potentially noisier views.

**Board position.** The peripheral group produced approximately 7.4% lower validation RMSE than the central group. Average normalized centroid distance was 0.323 versus 0.064. However, mean board hull area occupied 10.3% versus 23.6% of the image, and mean tilt differed (19.9° versus 16.9°). Image sharpness also varies. Therefore, this comparison supports collecting diverse positions but does not isolate position as the sole cause or establish a statistically significant effect.

**View angle.** The higher-tilt group (mean 35.8°) had approximately 54.8% lower validation RMSE than the low-tilt group (mean 9.0°). This is consistent with oblique views providing stronger constraints than nearly frontal views. The groups still differ in board size, position, and image quality; the result is observational, not a controlled causal estimate. It does not imply that arbitrarily steep angles are better: extreme tilt can impair corner localization.

**Limitations.** There are only 19 images and four fixed validation views. Validation extrinsics are fitted to the validation corners, so this assesses consistency under newly estimated poses rather than independently known 3D accuracy. The full-data calibration supplies approximate angle labels, including information from held-out images; experimental intrinsics are fitted exclusively to training images, but the angle grouping is not fully independent. Position and angle groups have only one run each. Blur, screen display artifacts, board size, and lighting were not independently controlled. No ground-truth camera calibration or metric accuracy measurement is available. Full-dataset RMSE and held-out RMSE use different image sets and should not be directly ranked as interchangeable scores.

**Practical recommendation.** Use multiple clear views with both frontal and moderate oblique angles, distribute the board across the frame, and retain consistent focus, focal length, and image geometry. Additional blurred or redundant photographs may be less useful than fewer diverse, sharp views.

## 3D visualization

```powershell
python camera_calibration/export_ply.py --calibration camera_calibration/results/calibration.json
```

Import `camera_calibration/results/calibration_scene.ply` into MeshLab and enable point rendering. Black/white points represent the board, red points represent camera centers, and blue lines indicate frusta.

## Static sphere AR and combined mesh

```powershell
python camera_calibration/project_sphere.py --images assignment3 --calibration camera_calibration/results/calibration.json --output camera_calibration/results/ar_sphere --radius 0.5
```

This saves projected photographs, `sphere_gallery.jpg`, and `calibration_scene_with_sphere.ply`. The combined triangle mesh contains the board, a blue sphere of radius half a square edge, red camera centers, blue frusta, and green optical-axis indicators. The sphere touches the board and lies on the camera-facing side. These are virtual visualization geometries, not reconstructed objects. Import the combined PLY into MeshLab to view it in 3D.

## Real-time Basler demonstration

```powershell
python camera_calibration/live_sphere.py
```

The program uses pypylon continuous latest-frame acquisition, converts frames to BGR, and reads existing intrinsics/distortion from `camera_calibration/results/calibration.json`. It detects the board, estimates pose with `solvePnP`, and draws the sphere on each original distorted frame. If the board is not fully detected, the sphere is not drawn.

- Use the same calibrated camera, unchanged focus/focal length and ROI/binning. Close other applications holding the camera connection.
- The default requested resolution is 2592 × 1944. Basler dimensions must match the requested dimensions. Same-aspect-ratio resizing scales intrinsics, but this is valid only for resizing, not sensor cropping or a changed optical setup.
- With multiple cameras, select the calibrated camera using `--serial SERIAL_NUMBER` or `--camera 1`.
- Optional fixed exposure/gain example: `--exposure-us 10000 --gain-db 0`. Adapt these values to lighting; unspecified values retain the camera's current values, with auto exposure/gain disabled.
- **Q/Esc** quits, **S** saves a snapshot, and **R** toggles recording in `camera_calibration/results/live_demo/`.
- AVI recording uses a fixed 15 FPS, so playback duration may differ from real elapsed time if actual acquisition/rendering FPS differs.

Offline validation, without connecting a camera:

```powershell
python camera_calibration/live_sphere.py --check-images assignment3
```

All 19 images passed detection, pose estimation, and rendering in the prior offline check. This is not confirmation of live camera connectivity or real-time frame rate. `offline_validation.json` logs each image's tracking status and pose reprojection error.

## GitHub upload notes

Keep scripts, this README, dependencies, calibration data, and experiment records. The 19 original BMP images total approximately 287 MB (decimal); consider Git LFS or distributing the image dataset separately. If excluding images, document how to obtain them so experiments remain reproducible. Live recordings and transient Python files are excluded by `.gitignore`.

No repository has been published by these scripts. A license is not selected automatically. Cite OpenCV, Basler/pypylon, any reused code, and AI assistance as required by the course policy. The Basler connection workflow was adapted from the team's `lab1/Lab1_Group4/Code/grab_basler1.1.py`.
