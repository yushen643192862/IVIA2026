# Lab1 Report

> Group number: 4. The Basic Part retains the recommended outline and requires additional content. The Bonus Part is provided below.

## Recommended Submission Structure

The report may follow this outline or use another structure. The results and analysis are the main focus.

```text
Lab1_Group4/
├── Images/
├── Code/                   # Include if code was used
└── Lab1_Report.pdf
```

## 1. Basic Part

### 1.1 Introduction

To be completed: briefly describe the purpose of the experiment.

### 1.2 Experiment Setup

To be completed: describe the camera parameter values and other experimental settings.

### 1.3 Result and Data Processing

To be completed: present the captured images and explain the image and data processing methods used for analysis.

### 1.4 Analysis and Discussion

To be completed: analyze and discuss the results using graphs, equations, and other supporting evidence.

### 1.5 Conclusion

To be completed: summarize the conclusions of the basic experiment.

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
