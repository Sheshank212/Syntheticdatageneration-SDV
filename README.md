# Synthetic Data Generation for COVID-19 & Cardiovascular Data using CTGAN & TVAE

## Overview

This project applies deep generative models, specifically **CTGAN** and **TVAE**, from the SDV library to generate high-fidelity synthetic healthcare datasets. The goals are to:

* Preserve patient privacy
* Enable model training without real data exposure
* Analyze prediction consistency using SHAP

Two public datasets are used:

* COVID-19 patient records (Mexican government via Kaggle)
* UCI Heart Disease dataset

## Objectives

* Preprocess real-world medical datasets with missing and inconsistent values
* Generate realistic synthetic datasets using:

  * CTGAN (Conditional Tabular GAN)
  * TVAE (Tabular Variational Autoencoder)
  * GaussianCopula (comparison baseline)
* Evaluate synthetic data using:

  * KDE plots
  * Bar plots for categorical columns
  * SHAP interpretability
* Use Random Forest models to test prediction quality
* Compare statistical summaries and feature influence

## Datasets

### COVID-19 Dataset

* \~1 million records, downsampled to 1000 for experimentation
* Comorbidities: COPD, Asthma, Hypertension, Cardiovascular, etc.
* Target: ICU admission
* Preprocessing:

  * Replaced \[97, 98, 99] with NaNs
  * Imputed numerical NaNs with median
  * Clipped Age between 1 and 100

### UCI Heart Disease Dataset

* Attributes: Cholesterol, Age, Blood Pressure, etc.
* Target: Heart disease diagnosis

## Methodology

### Preprocessing

* Uniform missing value treatment
* Label encoding for SHAP & model training
* Separate preprocessing pipelines for COVID and Cardio data

### Synthetic Data Generation

* **CTGAN**: trained for 50 epochs with batch size 1000
* **TVAE**: trained separately and compared with CTGAN
* **CopulaGAN**: included for AGE column comparison

### Evaluation Techniques

* **Statistical Summaries**: mean, std, min, max
* **KDE & Bar Plots**: Compare real vs synthetic distributions
* **SHAP Analysis**:

  * Ran Random Forest classifiers on synthetic data
  * Visualized feature importance for ICU prediction

## Visual Output Examples

* `AGE_distribution_comparison.png`
* `ICU_categorical_comparison.png`
* `PREGNANT_categorical_comparison.png`
* `shap_ctgan.png`, `shap_tvae.png`
* `loss_ctgan.png`, `loss_tvae.png`

## Key Observations

* CTGAN performed better on preserving distribution structure
* SHAP values for CTGAN highlighted **INTUBED**, **PNEUMONIA**, and **AGE**
* TVAE captured broader trends, sometimes underweighted expected comorbidities

## File Structure

```
Repo1/
├── datasets/
│   ├── Covid_Data.csv
│   └── heart_disease_dataset.csv
├── ctgan_tvae_covid_cardio.py
├── results/
│   ├── *.png
├── Covid_Synthetic_Data_CTGAN.csv
├── Covid_Synthetic_Data_TVAE.csv
├── Heart_Synthetic_Data_CTGAN.csv
├── Heart_Synthetic_Data_TVAE.csv
├── README.md
├── requirements.txt
└── .gitignore
```

## Tools Used

* Python, Pandas, NumPy
* SDV (CTGAN, TVAE, GaussianCopula)
* SHAP, LIME (exploratory)
* Plotly, Seaborn, Matplotlib
* Scikit-learn (Random Forest)

## Authors

* Sheshank Priyadarshi
* Parth Chopra

## License

MIT
