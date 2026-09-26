# Lab1 Report

> Group number: 4.

## Recommended Submission Structure

The report may follow this outline or use another structure. The results and analysis are the main focus.

```text
Lab1_Group4/
├── Images/
├── Code/
└── Lab1_Report.pdf
```

## 1. Basic Part

### 1.1 Introduction

The purpose of this experiment is to learn the operation and image acquisition workflow of the Basler camera using the pylon SDK. Furthermore, it aims to quantitatively investigate how varying exposure time and gain influence fundamental image properties (such as luminance and color saturation) and to establish mathematical models describing their relationships.

### 1.2 Experiment Setup

* **Hardware & Software Environment:**
  * **Camera Model:** Basler daA2500-14uc (connected via USB 3.0 interface).
  * **Software Platform:** Basler pylon Viewer (Developer Mode) and Python image acquisition scripts.
  * **Target Scene:** A uniformly illuminated planar surface (white wall) under steady ambient illumination to minimize spatial lighting variations.

* **Camera Configuration:**
  * Automatic exposure was turned off (`Exposure Auto = Off`).
  * Automatic gain was disabled (`Gain Auto = Off`).
  * Gamma correction was set to 1.0 to ensure linear sensor response.
  * The lens focus and aperture were locked during testing to keep optical throughput strictly constant.

* **Sampling Parameters:**
  * **Exposure Time ($e$):** Sampled across 29 discrete exposure levels spanning from $10\,\mu\text{s}$ to $990{,}000\,\mu\text{s}$.
  * **Gain ($g$):** Sampled across 24 discrete gain settings ranging from $1\text{ dB}$ up to $24\text{ dB}$.
  * In total, $29 \times 24 = 696$ image frames were captured across the complete parameter space.


### 1.3 Result and Data Processing

![Exposure-Gain Grid](/IVIA2026/Images/exposure_gain_grid.png)
*Figure 1: Two-dimensional image matrix across all sampled exposure times and gains (696 frames).*

#### 1. Parameter Grid Visualization
To comprehensively assess the imaging behavior across the entire parameter space, all $696$ captured images ($29$ exposure times $\times$ $24$ gain levels) were assembled into a two-dimensional image matrix (Figure 1). 
* The **horizontal axis** represents increasing exposure time from left to right ($10\,\mu\text{s}$ to $990{,}000\,\mu\text{s}$).
* The **vertical axis** represents increasing gain from top to bottom ($1\text{ dB}$ to $24\text{ dB}$).

As visually demonstrated across the grid:
* **Left Columns:** The images suffer from severe underexposure due to insufficient exposure time, resulting in near-zero pixel intensities.
* **Right Columns & Lower-Right Area:** High exposure durations combined with elevated gains drive the sensor into deep saturation, clipping pixel intensities at the 8-bit ceiling ($255$).
* **Central Transition Region:** The scene details remain well-preserved within the camera's linear dynamic range without significant under- or over-exposure clipping.

#### 2. Data Screening & Feature Extraction
The image grid clearly highlights the necessity of data screening: including heavily clipped (pure black or saturated white) samples in regression would distort both photometric curves and noise estimations. Therefore, based on mean intensity thresholds and saturation limits, valid frames within the linear dynamic range were retained for quantitative feature extraction.

![Noise and Brightness ROI](/IVIA2026/Images/Analysis/noise_roi.png)
*Figure 2: Selected Region of Interest (ROI) on the uniform white wall for brightness and noise estimation.*

For each accepted image, a fixed Region of Interest (ROI) spanning $[x: 682 \text{ to } 889,\, y: 415 \text{ to } 830]$ on the uniform white wall was extracted across all frames (Figure 2) to calculate:
1. **Average Grayscale Luminance ($\bar{Y}$):** The ROI was converted to single-precision float32 and transformed into a grayscale representation using the standard ITU-R luminance weights ($Y = 0.299R + 0.587G + 0.114B$), after which the spatial mean intensity was computed.
2. **Spatial Noise Standard Deviation ($\hat{\sigma}$):** The high-frequency noise residual was extracted via $N = I_{\text{ROI}} - \text{GaussianBlur}(I_{\text{ROI}})$ with a $5 \times 5$ kernel ($\sigma = 1.0$), and its sample standard deviation ($\text{ddof} = 1$, excluding boundary pixels) was calculated to quantify noise intensity.

To be completed: present the captured images and explain the image and data processing methods used for analysis.

### 1.4 Analysis and Discussion

![Comparison](/IVIA2026/Images/Analysis/basic_gain_curves.png)
*Figure 3: Quantitative responses across varying sensor gains: (left) Saturation $S$ and Value $V$ response curves (gain-s/v); (right) Grayscale luminance $Y$ response curve (gain-y).   

#### 1. Mathematical Formulations

The observed curves are governed by the camera hardware's analog signal amplification and standard color space transformations:

* **Analog Voltage Gain ($A$):**
  Sensor gain $g$ is configured in decibels (dB) in the acquisition metadata. The physical amplification factor $A$ applied to photodiode charges follows the logarithmic definition:
  $$A = 10^{\frac{g}{20}}$$

* **Grayscale Luminance ($Y$):**
  Luminance is calculated using the standard ITU-R BT.601 weighted sum implemented in the extraction pipeline:
  $$Y = 0.299R + 0.587G + 0.114B$$

* **HSV Value ($V$) and Saturation ($S$):**
  Per the standard definition of the HSV color model, Value reflects the peak channel intensity:
  $$V = \max(R, G, B)$$
  Saturation measures the purity of color relative to maximum intensity:
  $$S = \begin{cases} \frac{\max(R, G, B) - \min(R, G, B)}{\max(R, G, B)}, & \text{if } V \neq 0 \\ 0, & \text{if } V = 0 \end{cases}$$

---

#### 2. Quantitative Curve Analysis & Evidence

##### A. Luminance and Value Responses ($Y$ and $V$)
* **Exponential Amplification:** In Figure 3, both $V$ (left plot, orange line) and $Y$ (right plot, blue line) climb steadily from approximately $75$ to over $240$ as gain rises from $1\text{ dB}$ to $24\text{ dB}$. 
* **Nonlinear Acceleration:** In the low-to-medium gain region ($1\text{ dB} \le g \le 18\text{ dB}$), the response curves display an accelerating slope. This upward curvature directly reflects the exponential relationship $A = 10^{\frac{g}{20}}$, where equal step increases in decibels correspond to multiplying signal gains.
* **Saturation Clipping:** Near the highest gain settings ($g > 22\text{ dB}$), the slope begins to compress as pixel intensities approach the upper boundary of the 8-bit dynamic range ($255$).

##### B. Desaturation Effect ($S$)
* **Monotonic Decline:** As depicted in Figure 3 (left plot, blue line), saturation $S$ drops continuously from roughly $18$ down toward $3$.
* **Channel Compression:** Because the ROI corresponds to a painted white wall, baseline color saturation is naturally low. As analog amplification forces $R$, $G$, and $B$ channel readings toward the ceiling limit of $255$, the difference term $\max(R, G, B) - \min(R, G, B)$ contracts relative to $V$. This wash-out effect causes color information to degrade into clipped white, verifying that excessive sensor gain degrades chromatic fidelity.

### 1.5 Conclusion

In this experiment, the quantitative effects of camera exposure and electronic gain on image formation were systematically evaluated using the Basler daA2500-14uc sensor:

1. **Luminance and Gain Dynamics:** Sensor gain $g$ operates on a decibel scale, resulting in an exponential physical amplification ($A = 10^{g/20}$) of the photodiode signal. Consequently, both grayscale luminance $Y$ and HSV Value $V$ exhibit an accelerating nonlinear growth before reaching saturation near the 8-bit dynamic range ceiling ($255$).
2. **Chromatic Degradation:** Color saturation $S$ degrades monotonically under elevated gain levels. As intense analog amplification drives all primary channels ($R, G, B$) toward the upper clipping boundary, channel disparities vanish, inducing significant white wash-out and loss of color fidelity.
3. **Engineering Implications:** While increasing sensor gain effectively boosts image visibility in low-light conditions, it compresses dynamic range and destroys color purity. In practical computer vision and robotic acquisition pipelines, optical exposure time and physical aperture should be prioritized to maximize signal-to-noise ratio, reserving digital/electronic gain as a secondary measure.

## 2. Bonus Part

### 2.1 Objective and Modeling Approach

This experiment investigates the relationship between image noise intensity, exposure time $e$, and gain $g$. Since the specific functional relationship is unknown, polynomial regression is used to construct an empirical model. Model parameters are estimated from the experimental data, and the fitting performance of different polynomial degrees is compared. 

### 2.2 Data Acquisition and Screening

A total of **696 images were acquired using 29 exposure settings and 24 gain settings**. The images were first screened to exclude samples that were excessively dark, excessively bright, or contained a substantial proportion of saturated pixels. Near the lower or upper limits of the image intensity range, pixel fluctuations may be clipped, causing noise estimates to be artificially low. For example, a severely overexposed region may contain pixels with values close to 255 throughout. Even if the camera produces noise, these pixel values have little room to vary. Including such samples in the regression could lead to the misleading conclusion that noise decreases as exposure time increases. Screening was based on the mean intensity of a fixed white-wall region and the proportions of pixels near the black and white limits. **254 images were retained, and 442 images were excluded.**

### 2.3 Single-Image Noise Estimation

Only one image was captured for each exposure–gain combination. Therefore, temporal noise could not be measured directly from fluctuations at the same pixel across repeated frames. Instead, this experiment uses a **single-image spatial noise estimation method**. A white-wall region at the same location in every original image was selected as the region of interest (ROI), avoiding wall seams, object boundaries, and visible texture as far as possible. Although the wall is relatively uniform, its illumination may still vary gradually. Thus, the standard deviation of the ROI intensities was not used directly as the noise estimate. Instead, Gaussian smoothing was applied, and the residual between the original ROI and its smoothed version was calculated:

$$
N=I_{\mathrm{ROI}}-\operatorname{GaussianBlur}(I_{\mathrm{ROI}})
$$

The standard deviation of this residual was used as an approximate measure of image noise intensity:

$$
\hat{\sigma}=\operatorname{Std}(N)
$$

The same ROI and smoothing parameters were used for all images: a $5\times5$ Gaussian kernel with a standard deviation of 1. Border pixels affected by filtering were excluded when computing the residual statistics. 

### 2.4 Polynomial Regression Model

After feature extraction, each accepted image provided one data record:

$$
(e_i,\ g_i,\ \hat{\sigma}_i)
$$

Exposure time and gain were obtained from the actual values read back from the camera, in microseconds and dB, respectively. Using exposure time and gain as the independent variables and the estimated noise standard deviation as the dependent variable, a complete bivariate polynomial model was constructed:

$$
\hat{\sigma}=f(e,g)
=\sum_{\substack{i,j\geq0\\i+j\leq d}}c_{ij}e^ig^j
$$

Polynomial models of degrees 2, 3, 4, and 5 were compared, with 6, 10, 15, and 21 coefficients, respectively.

### 2.5 Training and Validation Results

To assess prediction performance on samples not used for fitting, the accepted data were split into **203 training samples and 51 validation samples** using a fixed random seed. The validation set accounted for approximately 20% of the accepted data. 
Performance was evaluated using mean absolute error (MAE), root mean squared error (RMSE), and the coefficient of determination ($R^2$).

| Model | Training RMSE | Validation MAE | Validation RMSE | Validation $R^2$ |
| --- | ---: | ---: | ---: | ---: |
| Degree 2 | 0.1641 | 0.1624 | 0.2143 | 0.6005 |
| Degree 3 | 0.1179 | 0.1179 | 0.1580 | 0.7829 |
| Degree 4 | 0.1014 | 0.0937 | **0.1401** | **0.8292** |
| Degree 5 | **0.0971** | **0.0901** | 0.1555 | 0.7897 |

The fitted fourth-degree equation, expressed in the original exposure and gain units, is:

$$
\begin{aligned}
\hat{\sigma}(e,g)\approx{}& 0.1517085119
+1.032140793\times10^{-4}e
+7.870890094\times10^{-2}g \\
&+6.067632465\times10^{-6}eg
-6.370232249\times10^{-9}e^2
-7.833887005\times10^{-3}g^2 \\
&+1.634077078\times10^{-13}e^3
-1.055638524\times10^{-10}e^2g
-2.494828333\times10^{-7}eg^2 \\
&+4.736397358\times10^{-4}g^3
-1.443352619\times10^{-18}e^4
+1.325540678\times10^{-15}e^3g \\
&-1.492670442\times10^{-11}e^2g^2
+1.774403796\times10^{-8}eg^3
-1.009040473\times10^{-5}g^4.
\end{aligned}
$$

Here, $e$ is in microseconds, $g$ is in dB, and $\hat{\sigma}$ is in gray levels. 

### 2.6 Analysis and Discussion

Both training and validation RMSE decreased as the polynomial degree increased from 2 to 4. This suggests that the additional polynomial terms better captured the relationships present in the current dataset. 

Increasing the degree to 5 further reduced training RMSE but increased validation RMSE, indicating signs of overfitting. Considering both validation squared error and model complexity, the **degree-4 model with 15 coefficients** was selected.
