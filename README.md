# Fronto-Thalamo-Cerebellar Circuitry in Predicting Cognition and Behavior of ABCD Adolescents

## Overview

This repository contains the analysis and findings from the study on the role of the **fronto-thalamo-cerebellar (FTC) circuitry** in predicting cognition and behavior in adolescents. Using data from the **Adolescent Brain Cognitive Development (ABCD)** study, this research highlights the predictive power of FTC gray and white matter features on neurocognitive and behavioral measures.

---

## Abstract

Attention deficit (AD) is a key dimension of mental health, presenting as a continuous trait across the general population and mental disorders. This study focuses on the FTC circuitry's structural variations and their role in predicting cognitive functions such as working memory and behavioral measures. Using **Bayesian Ridge Regression**, **Support Vector Regression**, and **Neural Networks**, the findings reveal the FTC's superior predictive capability for working memory over the traditionally studied **fronto-thalamo-parietal (FTP)** circuitry.

---

## Key Features

- **Dataset:** ABCD data release 5.1
- **Imaging Modalities:**
  - Structural MRI (sMRI)
  - Diffusion MRI (dMRI)
- **Circuitries Analyzed:**
  - Fronto-Thalamo-Cerebellar (FTC)
  - Fronto-Thalamo-Parietal (FTP)
  - Full Fronto-Thalamo-Parietal-Cerebellar (FTPC)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Materials and Methods](#materials-and-methods)
3. [Results](#results)
4. [Discussion and Conclusion](#discussion-and-conclusion)
5. [How to Use This Repository](#how-to-use-this-repository)
6. [References](#references)

---

## Introduction

Attention-deficit is a transdiagnostic symptom observed in disorders such as ADHD, autism spectrum disorder, and schizophrenia. This study leverages advanced imaging and predictive modeling techniques to investigate how FTC circuitry relates to neurocognition and behavior.

---

## Materials and Methods

### Preprocessing
1. **Structural MRI (sMRI):**
   - Processed using **SPM12** and **DARTEL** for gray matter volume (GMV) extraction.
2. **Diffusion MRI (dMRI):**
   - Processed with **FSL** and **ANTs** for fractional anisotropy (FA) feature extraction.

### Predictive Models
- **Bayesian Ridge Regression (BRR):**
  - Incorporates prior probability distributions for uncertainty estimation.
- **Support Vector Regression (SVR):**
  - Optimized with Bayesian hyperparameter search.
- **Neural Networks (NN):**
  - Two hidden layers with dropout for robustness.

### Evaluation
- Nested cross-validation with **5 outer folds** and **5 inner folds** ensures robust model evaluation.
- Metrics:
  - Coefficient of determination (r²)
  - Correlation between predicted and actual scores

---

## Results

The study highlights the FTC circuitry's superior performance in predicting working memory compared to FTP. The full FTPC circuitry demonstrates the highest predictive accuracy for most measures.

### Performance Summary
| Model | Circuitry | Target | r² | Correlation |
|-------|-----------|--------|----|-------------|
| BRR   | FTC       | c2b    | 0.052 ± 0.006 | 0.230 ± 0.013 |
| SVR   | FTPC      | c0b    | 0.062 ± 0.008 | 0.262 ± 0.018 |
| NN    | FTP       | CBCL   | 0.007 ± 0.005 | 0.084 ± 0.033 |

---

## Discussion and Conclusion

The findings emphasize the **cerebellum's contribution** to working memory and attention. This study serves as a foundation for further exploration of FTC circuitry and its implications for cognitive and behavioral outcomes in adolescents.

---
