
# 📄 Speech Understanding – Assignment 2 (CSL7770)

## 🔍 Overview

This repository contains the complete submission for **Assignment 2** of the course **CSL7770: Speech Understanding** offered at **IIT Jodhpur**, AY 2024-25 (Semester II).

### 👨‍🏫 Instructor
Professor Richa Singh

### 👤 Submitted by
- **Name**: Jyotishman Das  
- **Roll No**: M24CSA013  
- **Program**: M.Tech in AI, IIT Jodhpur

---

## 🧠 Question 1: Speech Enhancement (Q1-1 to Q1-4)

| Part | Task |
|------|------|
| Q1-1 | Pretrained and fine-tuned speaker verification using WavLM (LoRA + ArcFace) |
| Q1-2 | Multi-speaker dataset creation and SepFormer-based enhancement |
| Q1-3 | Speaker Identification of enhanced signals |
| Q1-4 | Novel enhanced pipeline combining SepFormer + Speaker ID model |

📁 Results and plots are saved in `results/` and `plots/` folders under `Speaker_separation` and `Enhanced_pipeline`.

Metrics used:
- EER, TAR@1%FAR, Accuracy (for speaker verification)
- SDR, SAR, SIR, PESQ (for enhancement evaluation)
- Rank-1 Accuracy (for identification)

---

## 🔤 Question 2: MFCC-Based Language Classification

- Dataset: Kaggle – Audio Dataset of 10 Indian Languages
- Selected Languages: **Hindi, Bengali, Tamil**
- Features extracted: **MFCCs using Librosa**
- Visualization: MFCC Spectrograms, Mean/Variance Stats
- Classification Model: **Random Forest**
- Accuracy Achieved: **~92%**

📁 All code in: `m24csa013-q2.ipynb`

---

## 🗂️ Folder Structure

M24CSA013 PA2/
├── Question 1/
│   ├── speaker_verification_final_full.py
│   ├── speech_enhancement_final.py
│   ├── enhanced_pipeline_final.py
│   ├── generate_plots.py
│   └── results/
│       ├── Speaker_verification/
│       │   ├── finetuned_results.csv
│       │   └── pretrained_results.csv
│       ├── Speaker_separation/
│       │   ├── evaluation_results.csv
│       │   ├── identification_mix_finetuned.csv
│       │   ├── identification_mix_pretrained.csv
│       │   └── plots/
│       │       ├── separation_metric_boxplot.png
│       │       ├── separation_kde_metrics.png
│       │       └── separation_metric_correlation.png
│       └── Enhanced_pipeline/
│           ├── enhanced_pipeline_evaluation_results.csv
│           ├── summary_Q1_4.txt
│           └── plots/
│               ├── Q1_4_boxplot_metrics.png
│               ├── Q1_4_corr_heatmap.png
│               └── Q1_4_kde_metrics.png
├── Question 2/
│   └── m24csa013-q2.ipynb
├── SUA2_Report.pdf
└── README.md

---

## ✅ Submission Guidelines Met

- [x] Python + PyTorch used
- [x] GitHub Repo Linked
- [x] PDF Report Attached
- [x] All results/plots included
- [x] Submitted Individually

---

## 📬 Contact
**Jyotishman Das**  
M.Tech AI | IIT Jodhpur  
GitHub: [rishi02102017](https://github.com/rishi02102017)
