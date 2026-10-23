# OSIC Pulmonary Fibrosis Progression — Super Ensemble

## Overview
This repository contains a machine learning super-ensemble model designed for the **OSIC Pulmonary Fibrosis Progression** challenge. The objective is to predict a patient's severity of decline in lung function—measured by Forced Vital Capacity ($\text{FVC}$)—using baseline CT scans and clinical metadata.

## Problem Statement
Pulmonary fibrosis is a disorder with no known cause and no known cure, created by scarring of the lungs. By accurately predicting $\text{FVC}$ decline along with a confidence measure ($\sigma$), this pipeline aims to aid clinical prognosis and personalized treatment analysis.

## Architecture & Methodology
* **Multimodal Data Processing:** Integrates tabular patient metadata (Age, Sex, SmokingStatus, Baseline FVC/Percent) with DICOM chest CT imaging features.
* **Super Ensemble Strategy:** Combines predictions across multiple model architectures (such as gradient boosted trees, tabular neural networks, and quantile regression models) to maximize stability and reduce variance.
* **Uncertainty Estimation:** Directly optimizes for both predicted $\text{FVC}$ and confidence intervals to handle heterogeneous patient progression rates.

## Evaluation Metric
Models are evaluated using a modified **Laplace Log Likelihood**, which penalizes both high prediction error and overconfident uncertainty estimates:

$$\sigma_{\text{clipped}} = \max(\sigma, 70)$$

$$\Delta = \min(|FVC_{\text{true}} - FVC_{\text{predicted}}|, 1000)$$

$$\text{Metric} = -\frac{\sqrt{2}\Delta}{\sigma_{\text{clipped}}} - \ln(\sqrt{2}\sigma_{\text{clipped}})$$

## Repository Structure
* `osic-pulmonary-fibrosis-progression-super-ensemb.ipynb` — Complete end-to-end notebook covering exploratory data analysis (EDA), feature engineering, model training, ensemble blending, and submission generation.

## Requirements & Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/samcamacho62-dotcom/OSIC-Pulmonary-Fibrosis-Progression---Super-Ensemb.git](https://github.com/samcamacho62-dotcom/OSIC-Pulmonary-Fibrosis-Progression---Super-Ensemb.git)
