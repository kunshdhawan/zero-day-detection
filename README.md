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
```

## Project Structure
<img width="500" height="400" alt="text-to-image" src="https://github.com/user-attachments/assets/4ad7385c-387e-462f-ad34-1a0340484c14" />


