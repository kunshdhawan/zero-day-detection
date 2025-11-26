# Zero-Day Network Intrusion & Anomaly Detection  
### Powered by Machine Learning (Isolation Forest) + CICIDS-2017

This project implements a **Zero-Day Attack Detector** using an unsupervised ML model  
(**Isolation Forest**) trained on the CICIDS-2017 dataset.  
The system learns patterns of **benign network flows** and detects unusual behavior  
indicating **zero-day intrusions, anomalies, and unseen attacks**.


## Features

- **Isolation Forest model** trained on real network flows  
- Detects **previously unseen (zero-day)** attacks  
- Accepts CSVs of network traffic and outputs:  
  - `NORMAL`  
  - `ANOMALY`  

- Interactive **Gradio web UI**
- Backend evaluation via `evaluate.py`
- ROC curve, PR curve, confusion matrix, anomaly score histogram
- Attack sample generator  
- Fully automated preprocessing & feature alignment


## Project Structure
zero-day-detection/
│
├── app.py # Gradio UI for predictions
├── evaluate.py # Advanced evaluation and metrics
├── train.py # Model training script
├── preprocess.py # Data cleaning & feature alignment
│
├── data/
│ ├── raw/ # Raw CICIDS CSVs
│ └── processed/
│   └── combined.csv # Cleaned megafile
│
├── models/
│ ├── isolation_forest.pkl # Trained model
│ └── feature_columns.json # EXACT 103 features model uses
│
├── results/
│ ├── confusion_matrix.png
│ ├── roc_curve.png
│ ├── precision_recall_curve.png
│ ├── anomaly_score_hist.png
│ └── evaluation_predictions.csv
│
└── README.md


## Concept

Unlike supervised classifiers that need labeled attack data,  
Zero-Day detection uses **unsupervised anomaly detection**:

1. Train model on **BENIGN** traffic only  
2. Model learns normal behavior  
3. Anything statistically unusual = **potential attack**

This allows detecting **new attacks**, even if unseen during training.


## Dataset

### Source: CICIDS-2017  
https://www.unb.ca/cic/datasets/ids-2017.html

We preprocess the flows into a unified CSV (`combined.csv`)  
and extract **103 numeric flow features** used during training.


## Installation

```bash
git clone <repo-url>
cd zero-day-detection

pip install -r requirements.txt
