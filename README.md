# ⛏️ RMR Tunnel Support Designer

A machine learning web application for real-time tunnel support design prediction based on Bieniawski's (1989) Rock Mass Rating (RMR89) system.

🔗 **Live App:** [Click here to try it](https://rmr-tunnel-support-designer-app-8tkfc2faxr48n4yhcnzhmx.streamlit.app/)

---

## 📌 Overview

This project implements and compares four machine learning models against a physics-based engine for predicting tunnel support requirements. Given 9 geotechnical inputs, the app predicts RMR score, rock mass class, bolt density, bolt length and shotcrete thickness in real time.

The support design equations extend beyond Bieniawski's basic Table 5 by incorporating span-dependent scaling, depth/stress corrections and excavation method adjustments — sourced from five peer-reviewed references.

---

## 🎯 Predictions

| Target | Description |
|--------|-------------|
| RMR Score | Rock Mass Rating (0–100) |
| Rock Mass Class | Class I to V (Bieniawski 1989) |
| Bolt Density | Bolts per m² |
| Bolt Length | metres |
| Shotcrete Thickness | mm (crown) |

---

## 🧠 Models

| Model | Strength |
|-------|----------|
| Ridge / Lasso + Logistic Regression | Strong baseline, best shotcrete for reference conditions |
| SVM / SVR (GridSearchCV) | Most accurate bolt length predictions |
| Random Forest (GridSearchCV) | Best physical consistency, correctly captures discrete class thresholds |
| ANN (PyTorch) | Closest RMR score predictions |

> **Key finding:** No single model dominates across all targets. Random Forest proved most reliable for discrete outputs while ANN delivered the most accurate RMR predictions. All models show increased uncertainty near RMR class boundaries — physically expected given the discrete nature of Bieniawski's classification.

---

## 📥 Input Features (9)

| Feature | Type | Range |
|---------|------|-------|
| UCS — Uniaxial Compressive Strength | Continuous | 1–500 MPa |
| RQD — Rock Quality Designation | Continuous | 0–100 % |
| Joint Spacing | Continuous | 0.01–5.0 m |
| Joint Condition | Discrete (1–5) | 1=best, 5=worst |
| Groundwater Condition | Discrete (1–5) | 1=dry, 5=flowing |
| Joint Orientation Adjustment | Discrete | 0, -2, -5, -10, -12 |
| Excavation Span | Continuous | 4–16 m |
| Overburden Depth | Continuous | 50–900 m |
| Excavation Method | Binary | 0=TBM, 1=Drill & Blast |

---

## 📐 Scientific Basis

Support equations sourced from five peer-reviewed references:

- **Bieniawski (1989)** — RMR89 classification system and Table 5 support guidelines
- **Lowson & Bieniawski (2013)** — Span-dependent bolt length design charts
- **Rehman et al. (2018)** — Shotcrete thickness as a function of tunnel span
- **Hoek & Marinos (2000)** — Depth/stress squeezing criterion
- **Barton et al. (1974)** — TBM vs Drill & Blast support corrections

---

## 📊 Dataset

- **1000 synthetic samples** — 200 per RMR class (stratified)
- **Measurement noise** applied to RMR89 parameters following:
  - Hack (2002) — UCS and Joint Spacing noise (σ = 12%)
  - Palmstrom (2005) — RQD noise (σ = 5%)
  - Cai et al. (2004) — Discrete parameter shifts (p = 20%)
  - Jakubec & Laubscher (2000) — Joint condition uncertainty
- Noise calibrated so a realistic fraction of samples cross RMR class boundaries
- `True_Rock_Class` retains the noiseless label; `Measured_Rock_Class` reflects noisy input

> ⚠️ **Limitation:** The models are trained on synthesized data. Although noise implementation follows established geotechnical noise models, synthesized data cannot fully replicate the complexity of real field conditions. Validation against field case studies remains an important next step.

---

## 🛠️ Tech Stack

```
Python          — Core language
Scikit-learn    — Ridge, Lasso, Logistic Regression, SVM/SVR, Random Forest
PyTorch         — ANN architecture and training
Streamlit       — Web application deployment
Pandas          — Data handling
NumPy           — Numerical computation
Scipy           — ANOVA F-tests
Matplotlib      — Visualisation
Seaborn         — Statistical plots
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/sayanmondal31694-sys/rmr-tunnel-support-designer.git
cd rmr-tunnel-support-designer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run rmr_app.py
```

---

## 📓 Notebooks

The full research pipeline is documented across three Jupyter notebooks:

| Notebook | Description |
|----------|-------------|
| [`rmr_dataset_generator.ipynb`](rmr_dataset_generator.ipynb) | Synthetic dataset generation — RMR89 rating functions, support design equations, measurement noise injection following Hack (2002), Palmstrom (2005), Cai et al. (2004) and Jakubec & Laubscher (2000) |
| [`rmr_eda.ipynb`](rmr_eda.ipynb) | Exploratory data analysis — 20 figures covering feature distributions, span/depth effects on support targets, TBM vs D&B comparison, noise-induced class boundary shifts, ANOVA F-tests and correlation analysis |
| [`rmr_model_training.ipynb`](rmr_model_training.ipynb) | Model training, hyperparameter tuning (GridSearchCV), architecture search (ANN), custom PyTorch k-fold CV, confusion matrices, actual vs predicted scatter plots and final comparative results |

---

## 📁 Repository Structure

```
rmr-tunnel-support-designer/
│
├── rmr_app.py                      # Streamlit web application
├── requirements.txt                # Python dependencies
├── .python-version                 # Python version (3.11)
│
├── rmr_dataset_generator.ipynb     # Notebook 1 — Dataset generation
├── rmr_eda.ipynb                   # Notebook 2 — Exploratory data analysis
├── rmr_model_training.ipynb        # Notebook 3 — Model training & evaluation
│
├── model_lr_rmr.pkl                # Ridge/Lasso — RMR
├── model_lr_class.pkl              # Logistic Regression — Rock Class
├── model_lr_bolt_density.pkl       # Ridge/Lasso — Bolt Density
├── model_lr_bolt_length.pkl        # Ridge/Lasso — Bolt Length
├── model_lr_shotcrete.pkl          # Ridge/Lasso — Shotcrete
│
├── model_svm_rmr.pkl               # SVR — RMR
├── model_svm_class.pkl             # SVC — Rock Class
├── model_svm_bolt_density.pkl      # SVR — Bolt Density
├── model_svm_bolt_length.pkl       # SVR — Bolt Length
├── model_svm_shotcrete.pkl         # SVR — Shotcrete
│
├── model_rf_rmr.pkl                # Random Forest — RMR
├── model_rf_class.pkl              # Random Forest — Rock Class
├── model_rf_bolt_density.pkl       # Random Forest — Bolt Density
├── model_rf_bolt_length.pkl        # Random Forest — Bolt Length
├── model_rf_shotcrete.pkl          # Random Forest — Shotcrete
│
├── scaler.pkl                      # StandardScaler (fitted on train set)
├── label_encoder.pkl               # LabelEncoder for Rock Class
├── ann_architectures.json          # ANN architecture config
└── ext_model_ann_*.pth             # PyTorch ANN weights
```

---

## 📚 References

1. Bieniawski, Z.T. (1989). *Engineering Rock Mechanics: Classification and Characterization.* Balkema, Rotterdam.
2. Lowson, A.R. & Bieniawski, Z.T. (2013). Critical Assessment of RMR-Based Tunnel Design Practices. *Proc. RETC 2013*, pp. 180–198.
3. Rehman, H., Naji, A.M., Kim, J. & Yoo, H. (2018). Empirical Evaluation of RMR and Q for Tunnel Support Design. *Applied Sciences, 8(5), 782.*
4. Hoek, E. & Marinos, P. (2000). Predicting Tunnel Squeezing. *Tunnels & Tunnelling International.*
5. Barton, N., Lien, R. & Lunde, J. (1974). Engineering classification of rock masses. *Rock Mechanics, 6(4), 189–236.*
6. Hack, R. (2002). An evaluation of slope stability classification. *ISRM Symposium.*
7. Palmstrom, A. (2005). Measurements of and correlations between block size and rock quality designation. *Tunnelling and Underground Space Technology, 20(4), 362–377.*
8. Cai, M., Kaiser, P.K., Uno, H., Tasaka, Y. & Minami, M. (2004). Estimation of rock mass strength and deformation modulus. *International Journal of Rock Mechanics, 41(3), 319–326.*
9. Jakubec, J. & Laubscher, D.H. (2000). The MRMR rock mass rating classification system in mining practice. *MassMin 2000*, Brisbane.

---

## 👤 Author

**Sayan Mondal**
- GitHub: [@sayanmondal31694-sys](https://github.com/sayanmondal31694-sys)

---

## ⭐ If you found this useful, please star the repository!
