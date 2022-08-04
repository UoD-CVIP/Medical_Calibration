# Calibration of Deep Medical Image Classifiers: An Empirical Comparison using Dermatology and Histopathology Datasets

**Jacob Carse, Andres Alvarez Olmo and Stephen McKenna**

CVIP, School of Science and Engineering, University of Dundee, Dundee, Scotland, UK


As deep learning classifiers become ever more widely deployed for medical image analysis tasks, issues of predictive calibration need to be addressed. Mis-calibration is the deviation between predictive probability (confidence) and classification correctness. Well-calibrated classifiers enable cost-sensitive and selective decision-making. This paper presents an empirical investigation of calibration methods on two medical image datasets (multi-class dermatology and binary histopathology image classification). We show the effect of temperature scaling with temperature optimized using various measures of calibration replacing the standard negative log-likelihood. We do so not only for networks trained using one-hot encoding and cross-entropy loss, but also using focal loss and label smoothing. We compare these with two Bayesian methods. Results suggest little or no advantage to the use of alternative calibration metrics for tuning temperature. Temperature scaling of networks trained using focal loss (with appropriate hyperparameters) provided strong results in terms of both calibration and accuracy across both datasets.


Trained models can be found [here](https://dmail-my.sharepoint.com/:u:/g/personal/jcarse_dundee_ac_uk/EUQwBB-0PApBqxPQZzO86vUB8xoVYhXEKjoiBcZ9F7mS9A?e=g1tngF).
