# Gait-Reorganization

Code and documentation for a research project focused on studying age-related reorganization of gait dynamics using full-body IMU data, preprocessing pipelines, gait-cycle segmentation, exploratory analysis, Dynamic Time Warping (DTW), and deep representation learning.

This repository contains scripts and notebooks for preparing gait data, detecting gait events, segmenting gait cycles, training autoencoder-based models, and exploring multivariate IMU-based gait kinematics collected during continuous overground walking.

The main objective is to investigate whether gait differences across the adult lifespan emerge as clearly separable groups or as a gradual reorganization of the locomotor manifold.

## Overview

Human gait is a high-dimensional behavior involving coordinated changes across many body segments and joints. Traditional spatiotemporal metrics such as gait speed, cadence, step length, and stride time are clinically useful, but they capture only a limited portion of locomotor organization.

This project explores gait as a multivariate system by combining:

- preprocessing of full-body IMU gait data
- gait-event detection and cycle segmentation
- exploratory data analysis
- waveform comparison using Dynamic Time Warping
- deep representation learning with autoencoders
- interpretable latent-space exploration

The repository is centered on the study of age-related gait organization using wearable full-body IMU recordings and cycle-level gait representations.

## Research Motivation

Aging modifies gait in subtle and distributed ways that are not always well described by isolated variables or simple group comparisons. Rather than treating the problem only as a classification task, this project investigates how aging shapes the structure of gait dynamics in latent space.

The central question is:

Do gait patterns across young, middle-aged, and older adults form distinct clusters, or do they reflect a gradual reorganization across the lifespan?

## Main Objectives

### 1. Preprocess multivariate gait signals
- organize trial-level IMU recordings
- detect gait events from foot-contact signals
- segment gait cycles
- normalize each cycle to a common temporal length
- standardize kinematic variables for downstream modeling

### 2. Learn compact latent representations
- train deep autoencoder models on cycle-level gait kinematics
- preserve waveform morphology while reducing dimensionality
- explore latent representations across model configurations

### 3. Explore waveform similarity
- compare gait cycles and kinematic waveforms using DTW
- inspect temporal alignment and waveform consistency
- support qualitative and quantitative comparisons across subjects and groups

### 4. Analyze latent-space organization
- inspect learned embeddings
- explore overlap and structure across age groups
- study subject-level and group-level organization in latent space

### 5. Support interpretable analysis
- summarize model outputs and experiments
- use permutation-based utilities to support downstream interpretability workflows
- prepare the repository for future expansion into broader explainability analyses

## Current Scope

This public version of the repository currently focuses on:

- preprocessing
- gait-event handling
- cycle segmentation
- exploratory data analysis
- Dynamic Time Warping
- autoencoder pipelines in PyTorch

Standalone nonlinear analysis modules are not yet included in this repository.

## Dataset Context

The project works with full-body IMU gait recordings collected during self-paced walking on a curved indoor track. The broader study design includes:

- three age groups
  - G01: young adults
  - G02: middle-aged adults
  - G03: older adults
- repeated continuous trials per subject
- full-body kinematic signals from wearable IMUs
- repeated measurements suitable for subject-level and group-level analysis

This repository mainly contains code and workflow utilities. 
Raw data may not be distributed directly here but can be found in: Wiles, T. et al. Nonan gaitprint: An imu gait database of healthy older adults. Figshare https://doi.org/10.6084/m9.figshare.27815034. (2024).

## Current Repository Structure

```text
Gait-Reorganization/
├── .gitignore
├── AE_pipeline_pytorch.py
├── Data_loader.py
├── DTW.ipynb
├── DTW.py
├── EDA.ipynb
├── gait_events.py
├── LSTM_AE.ipynb
├── perm_utils2.py
├── PP_pipeline.py
├── README.md
├── requirements.in
├── segment_utils.py
└── summary_utils.py

Key Features
Curved-track walking

Unlike many gait datasets acquired under treadmill or straight-line laboratory conditions, this project considers continuous overground walking on a curved indoor track, allowing richer ecological variability in gait behavior.

Long continuous recordings

Each trial contains long time series suitable for waveform-based modeling and cycle-level analysis.

Repeated acquisitions

Repeated trials across acquisition days support subject-level consistency analysis and longitudinal comparison.

Multivariate gait representation

The project uses full-body kinematics rather than isolated gait variables, allowing the analysis of distributed locomotor organization.

Representation learning

The repository includes autoencoder-based pipelines designed to learn compact latent representations of gait cycles while preserving biomechanical waveform structure.

Requirements
Python 3.10+
NumPy
Pandas
SciPy
scikit-learn
PyTorch
Matplotlib
Jupyter

Additional dependencies may be listed in requirements.in.

Getting Started

Clone the repository:

git clone https://github.com/carmare13/Gait-Reorganization.git
cd Gait-Reorganization

Create your environment and install dependencies according to your preferred workflow. For example, you may use pip-tools, pip, or a virtual environment manager of your choice based on requirements.in.

Project Status

This repository is under active development. It is being used to support ongoing research on gait dynamics, aging, preprocessing pipelines, temporal alignment analysis, and latent representation learning.

Citation

If you use this repository, please cite the corresponding paper or contact the author for the most appropriate citation format.

Contact

Diana C. Martinez
Email: dianacmartinez13@gmail.com
