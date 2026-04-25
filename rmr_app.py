"""
RMR Tunnel Support Design — Streamlit App
==========================================
Predicts: RMR Score · Rock Class · Bolt Density · Bolt Length · Shotcrete Thickness
Models  : Ridge/Lasso · Logistic Regression · SVM/SVR · Random Forest · ANN (PyTorch)
Inputs  : 6 RMR89 params (Bieniawski 1989) + Excavation Span · Depth · Excavation Method
"""

import streamlit as st
import numpy as np
import json
import os

# ── Optional heavy imports (graceful degradation if models not present) ─────
try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RMR Tunnel Support Designer",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .stApp { background: #0f1117; }

  /* Sidebar */
  [data-testid="stSidebar"] { background: #1a1d27 !important; }

  /* Cards */
  .metric-card {
    background: #1e2130;
    border: 1px solid #2d3148;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    text-align: center;
    height: 100%;
  }
  .metric-card .label  { font-size: 0.75rem; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
  .metric-card .value  { font-size: 2rem; font-weight: 700; color: #e6edf3; }
  .metric-card .unit   { font-size: 0.8rem; color: #8892b0; margin-top: 2px; }

  /* Class badge */
  .class-badge {
    display: inline-block;
    padding: 0.35rem 1.2rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 1px;
  }

  /* Section titles */
  .section-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #58a6ff;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 1.2rem 0 0.4rem 0;
    border-bottom: 1px solid #21262d;
    padding-bottom: 4px;
  }

  /* RMR progress bar wrapper */
  .rmr-bar-outer {
    background: #21262d;
    border-radius: 8px;
    height: 22px;
    width: 100%;
    overflow: hidden;
    margin: 6px 0;
  }
  .rmr-bar-inner {
    height: 100%;
    border-radius: 8px;
    transition: width 0.4s ease;
  }

  /* Reference table */
  .ref-table { font-size: 0.78rem; width: 100%; border-collapse: collapse; }
  .ref-table th { color: #58a6ff; font-weight: 600; padding: 4px 8px; border-bottom: 1px solid #30363d; text-align: left; }
  .ref-table td { padding: 4px 8px; color: #c9d1d9; border-bottom: 1px solid #21262d; }

  /* Formula box */
  .formula-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-family: monospace;
    font-size: 0.82rem;
    color: #79c0ff;
    margin: 6px 0;
  }

  /* Warning / info banners */
  .info-banner {
    background: #0d2a4a;
    border-left: 4px solid #58a6ff;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 1rem;
    font-size: 0.82rem;
    color: #c9d1d9;
    margin: 8px 0;
  }
  .warn-banner {
    background: #2d1f00;
    border-left: 4px solid #d29922;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 1rem;
    font-size: 0.82rem;
    color: #c9d1d9;
    margin: 8px 0;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS / REFERENCE ENGINE  (mirrors rmr_dataset_generator.ipynb exactly)
# ─────────────────────────────────────────────────────────────────────────────

CLASS_ORDER  = ['I', 'II', 'III', 'IV', 'V']
CLASS_COLORS = {'I': '#2ecc71', 'II': '#3498db', 'III': '#f39c12',
                'IV': '#e67e22', 'V': '#e74c3c'}

JC_MAP     = {1: 30, 2: 25, 3: 20, 4: 10, 5: 0}
GW_MAP     = {1: 15, 2: 10, 3: 7,  4: 4,  5: 0}
ORIENT_ADJ = [0, -2, -5, -10, -12]

BOLT_DENSITY_MAP = {'I': 0.00, 'II': 0.16, 'III': 0.44, 'IV': 1.00, 'V': 1.78}


def ucs_rating(u):
    if   u > 250: return 15
    elif u > 100: return 12
    elif u > 50:  return 7
    elif u > 25:  return 4
    elif u > 5:   return 2
    else:         return 1

def rqd_rating(r):
    if   r >= 90: return 20
    elif r >= 75: return 17
    elif r >= 50: return 13
    elif r >= 25: return 8
    else:         return 3

def js_rating(s):
    if   s > 2:    return 20
    elif s > 0.6:  return 15
    elif s > 0.2:  return 10
    elif s > 0.06: return 8
    else:          return 5

def compute_rmr(ucs, rqd, js, jc, gw, orient_adj):
    return ucs_rating(ucs) + rqd_rating(rqd) + js_rating(js) + JC_MAP[jc] + GW_MAP[gw] + orient_adj

def get_class(rmr):
    if   rmr >= 81: return 'I'
    elif rmr >= 61: return 'II'
    elif rmr >= 41: return 'III'
    elif rmr >= 21: return 'IV'
    else:           return 'V'

def depth_stress_factor(rmr, depth_m, ucs_mpa):
    sigma_v  = 0.027 * depth_m
    sigma_cm = 0.5 * ucs_mpa * np.exp((rmr - 100) / 24.0)
    sigma_cm = max(sigma_cm, 0.5)
    ratio    = sigma_v / sigma_cm
    return min(1.5, 1.0 + 0.3 * max(0.0, ratio - 0.2))

def base_bolt_length(rmr):
    if   rmr >= 81: return 2.0
    elif rmr >= 61: return 3.0
    elif rmr >= 41: return 4.0
    elif rmr >= 21: return 4.5
    else:           return 5.5

def span_fraction(rmr):
    if   rmr >= 61: return 0.30
    elif rmr >= 41: return 0.40
    elif rmr >= 21: return 0.50
    else:           return 0.60

def compute_bolt_length(rmr, span, depth, ucs, method):
    lb = max(base_bolt_length(rmr), span_fraction(rmr) * span)
    if method == 0:
        lb *= 0.90
    df  = depth_stress_factor(rmr, depth, ucs)
    lb *= 1.0 + 0.5 * (df - 1.0)
    return float(np.clip(lb, 2.0, 7.0))

def base_shotcrete_mm(rmr):
    if   rmr >= 81: return 0.0
    elif rmr >= 61: return 25.0
    elif rmr >= 41: return 75.0
    elif rmr >= 21: return 125.0
    else:           return 175.0

def compute_shotcrete_mm(rmr, span, depth, ucs, method):
    t = base_shotcrete_mm(rmr) * (span / 10.0)
    if method == 0:
        t *= 0.80
    df = depth_stress_factor(rmr, depth, ucs)
    t *= df
    return float(np.clip(t, 0.0, 250.0))

def physics_predict(ucs, rqd, js, jc, gw, orient_adj, span, depth, method):
    rmr          = compute_rmr(ucs, rqd, js, jc, gw, orient_adj)
    rock_class   = get_class(rmr)
    bolt_density = BOLT_DENSITY_MAP[rock_class]
    bolt_length  = compute_bolt_length(rmr, span, depth, ucs, method)
    shotcrete    = compute_shotcrete_mm(rmr, span, depth, ucs, method)
    return {
        'rmr': rmr, 'class': rock_class,
        'bolt_density': bolt_density,
        'bolt_length':  bolt_length,
        'shotcrete':    shotcrete,
    }

def get_individual_ratings(ucs, rqd, js, jc, gw, orient_adj):
    return {
        'R1 UCS':           ucs_rating(ucs),
        'R2 RQD':           rqd_rating(rqd),
        'R3 Joint Spacing': js_rating(js),
        'R4 Joint Cond.':   JC_MAP[jc],
        'R5 Groundwater':   GW_MAP[gw],
        'B  Orientation':   orient_adj,
    }

# ─────────────────────────────────────────────────────────────────────────────
# ANN MODEL CLASS
# ─────────────────────────────────────────────────────────────────────────────
if TORCH_OK:
    class RMR_ANN(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_sizes=[64, 32], dropout=0.2):
            super().__init__()
            layers, prev = [], input_dim
            for h in hidden_sizes:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, output_dim))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Try to load saved model files. Return dict of what's available."""
    models = {}

    # Sklearn models
    if JOBLIB_OK:
        sklearn_files = {
            'scaler':   'scaler.pkl',
            'encoder':  'label_encoder.pkl',
            'lr_rmr':   'model_lr_rmr.pkl',
            'lr_cls':   'model_lr_class.pkl',
            'lr_bd':    'model_lr_bolt_density.pkl',
            'lr_bl':    'model_lr_bolt_length.pkl',
            'lr_sc':    'model_lr_shotcrete.pkl',
            'svm_rmr':  'model_svm_rmr.pkl',
            'svm_cls':  'model_svm_class.pkl',
            'svm_bd':   'model_svm_bolt_density.pkl',
            'svm_bl':   'model_svm_bolt_length.pkl',
            'svm_sc':   'model_svm_shotcrete.pkl',
            'rf_rmr':   'model_rf_rmr.pkl',
            'rf_cls':   'model_rf_class.pkl',
            'rf_bd':    'model_rf_bolt_density.pkl',
            'rf_bl':    'model_rf_bolt_length.pkl',
            'rf_sc':    'model_rf_shotcrete.pkl',
        }
        for key, fname in sklearn_files.items():
            if os.path.exists(fname):
                try:
                    models[key] = joblib.load(fname)
                except Exception:
                    pass

    # ANN
    if TORCH_OK and os.path.exists('ann_architectures.json'):
        try:
            with open('ann_architectures.json') as f:
                arch_info = json.load(f)
            models['ann_arch'] = arch_info

            task_spec = {
                'RMR':         ('ann_rmr',   1),
                'Class':       ('ann_cls',   5),
                'BoltDensity': ('ann_bd',    1),
                'BoltLength':  ('ann_bl',    1),
                'Shotcrete':   ('ann_sc',    1),
            }
            for task_name, (key, out_dim) in task_spec.items():
                # Try both naming conventions from notebooks
                for fname_template in [
                    f'ext_model_ann_{task_name.lower()}.pth',
                    f'model_ann_{task_name.lower()}.pth',
                ]:
                    if os.path.exists(fname_template):
                        hidden = arch_info.get(task_name, [64, 32])
                        m = RMR_ANN(9, out_dim, hidden)
                        m.load_state_dict(torch.load(fname_template, map_location='cpu'))
                        m.eval()
                        models[key] = m
                        break
        except Exception:
            pass

    return models

# ─────────────────────────────────────────────────────────────────────────────
# ML PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def ml_predict(models, X_raw, model_key_prefix):
    """Run ML prediction. Returns dict matching physics_predict output, or None."""
    if not models:
        return None

    scaler  = models.get('scaler')
    encoder = models.get('encoder')

    # Decide if this family uses scaled input
    needs_scale = model_key_prefix in ('lr', 'svm', 'ann')

    X = np.array(X_raw, dtype=np.float64).reshape(1, -1)

    if needs_scale and scaler:
        X_sc = scaler.transform(X)
    else:
        X_sc = X   # RF uses raw

    results = {}
    tasks = [
        ('rmr',          'rmr',  'regression'),
        ('cls',          'cls',  'classification'),
        ('bd',           'bd',   'regression'),
        ('bl',           'bl',   'regression'),
        ('sc',           'sc',   'regression'),
    ]

    for short, res_key, task_type in tasks:
        key  = f'{model_key_prefix}_{short}'
        ann_key = {
            'lr_rmr': None, 'svm_rmr': None, 'rf_rmr': None,
            'ann_rmr': 'ann_rmr', 'ann_cls': 'ann_cls',
            'ann_bd': 'ann_bd', 'ann_bl': 'ann_bl', 'ann_sc': 'ann_sc',
        }.get(key, None)

        model = models.get(key)
        if model is None:
            return None

        try:
            if model_key_prefix == 'ann' and TORCH_OK:
                with torch.no_grad():
                    t = torch.tensor(X_sc, dtype=torch.float32)
                    out = model(t)
                    if task_type == 'classification':
                        pred = torch.argmax(out, dim=1).item()
                    else:
                        pred = out.squeeze().item()
            else:
                pred = model.predict(X_sc if needs_scale else X)[0]

            results[res_key] = pred
        except Exception:
            return None

    if encoder and 'cls' in results:
        cls_idx = int(round(results['cls']))
        cls_idx = max(0, min(cls_idx, len(CLASS_ORDER) - 1))
        results['class'] = encoder.inverse_transform([cls_idx])[0]
    else:
        results['class'] = get_class(int(round(results.get('rmr', 50))))

    return {
        'rmr':          results.get('rmr', 0),
        'class':        results['class'],
        'bolt_density': results.get('bd', 0),
        'bolt_length':  results.get('bl', 0),
        'shotcrete':    results.get('sc', 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUPPORT DESCRIPTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
CLASS_DESC = {
    'I':   ('Very Good Rock', '#2ecc71', 'Rock is self-supporting. Spot bolts where needed. No systematic shotcrete required.'),
    'II':  ('Good Rock',      '#3498db', 'Systematic bolting and thin shotcrete in crown. Occasional wire mesh.'),
    'III': ('Fair Rock',      '#f39c12', 'Systematic bolting with steel arches. 50–100 mm shotcrete in crown and sides.'),
    'IV':  ('Poor Rock',      '#e67e22', 'Systematic bolting, steel arches, and 100–150 mm shotcrete. Forepoling may be needed.'),
    'V':   ('Very Poor Rock', '#e74c3c', 'Heavy systematic support, steel arches, and ≥150 mm shotcrete. Immediate support required.'),
}

SUPPORT_TABLE_HTML = """
<table class="ref-table">
  <tr><th>Class</th><th>RMR</th><th>Description</th><th>Bolts</th><th>Shotcrete</th><th>Steel Arches</th></tr>
  <tr><td><b style="color:#2ecc71">I</b></td><td>81–100</td><td>Very Good</td><td>Spot bolts</td><td>None</td><td>None</td></tr>
  <tr><td><b style="color:#3498db">II</b></td><td>61–80</td><td>Good</td><td>2.5m spacing</td><td>0–50 mm</td><td>None</td></tr>
  <tr><td><b style="color:#f39c12">III</b></td><td>41–60</td><td>Fair</td><td>1.5m spacing</td><td>50–100 mm</td><td>Light</td></tr>
  <tr><td><b style="color:#e67e22">IV</b></td><td>21–40</td><td>Poor</td><td>1.0m spacing</td><td>100–150 mm</td><td>Medium</td></tr>
  <tr><td><b style="color:#e74c3c">V</b></td><td>&lt;21</td><td>Very Poor</td><td>0.75m spacing</td><td>150–200 mm</td><td>Heavy</td></tr>
</table>
"""

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — INPUT PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⛏️ RMR Parameters")

    st.markdown('<div class="section-title">RMR89 Ratings (Bieniawski 1989)</div>', unsafe_allow_html=True)

    ucs = st.number_input(
        "UCS — Uniaxial Compressive Strength (MPa)",
        min_value=1.0, max_value=500.0, value=80.0, step=5.0,
        help="Intact rock strength. Drives rating R1 (max 15 pts)."
    )
    rqd = st.slider(
        "RQD — Rock Quality Designation (%)",
        min_value=0, max_value=100, value=75,
        help="Drill core recovery ratio. Rating R2 (max 20 pts)."
    )
    js = st.number_input(
        "Joint Spacing (m)",
        min_value=0.01, max_value=5.0, value=0.8, step=0.05,
        help="Mean spacing between joints. Rating R3 (max 20 pts)."
    )
    jc = st.selectbox(
        "Joint Condition (1=best → 5=worst)",
        options=[1, 2, 3, 4, 5],
        index=1,
        format_func=lambda x: {
            1: "1 — Very rough, tight, unweathered",
            2: "2 — Slightly rough, separation <1mm",
            3: "3 — Slightly rough, separation >1mm",
            4: "4 — Slickensided / gouge <5mm",
            5: "5 — Soft gouge >5mm / open joint",
        }[x],
        help="Joint surface condition. Rating R4 (max 30 pts)."
    )
    gw = st.selectbox(
        "Groundwater Condition (1=best → 5=worst)",
        options=[1, 2, 3, 4, 5],
        index=0,
        format_func=lambda x: {
            1: "1 — Completely dry",
            2: "2 — Damp",
            3: "3 — Wet",
            4: "4 — Dripping",
            5: "5 — Flowing",
        }[x],
        help="Groundwater inflow. Rating R5 (max 15 pts)."
    )
    orient_idx = st.selectbox(
        "Joint Orientation Adjustment",
        options=[0, 1, 2, 3, 4],
        index=0,
        format_func=lambda x: {
            0: "0  — Very favourable",
            1: "-2 — Favourable",
            2: "-5 — Fair",
            3: "-10 — Unfavourable",
            4: "-12 — Very unfavourable",
        }[x],
        help="Strike/dip orientation penalty (Table 3, Bieniawski 1989)."
    )
    orient_adj = ORIENT_ADJ[orient_idx]

    st.markdown('<div class="section-title">Excavation Context</div>', unsafe_allow_html=True)

    span = st.slider(
        "Excavation Span (m)",
        min_value=3.0, max_value=20.0, value=10.0, step=0.5,
        help="Tunnel width / span. Support scales with span (Rehman et al. 2018)."
    )
    depth = st.slider(
        "Overburden Depth (m)",
        min_value=50, max_value=1000, value=200, step=25,
        help="Depth below surface. Drives stress modifier (Hoek & Marinos 2000). Bieniawski valid <926 m."
    )
    method = st.radio(
        "Excavation Method",
        options=[0, 1],
        format_func=lambda x: "TBM (Tunnel Boring Machine)" if x == 0 else "Drill & Blast (D&B)",
        help="TBM produces smoother profile → 10% less bolt length, 20% less shotcrete."
    )

    st.markdown('<div class="section-title">Prediction Model</div>', unsafe_allow_html=True)
    models = load_models()
    has_sklearn = 'scaler' in models
    has_ann     = 'ann_rmr' in models

    model_options = ["Physics Engine (Bieniawski 1989)"]
    if has_sklearn:
        model_options += ["Ridge / Lasso + LogReg", "SVM / SVR", "Random Forest"]
    if has_ann:
        model_options.append("ANN (PyTorch)")

    selected_model = st.selectbox("Select Prediction Model", model_options)

    if not has_sklearn and len(model_options) == 1:
        st.markdown("""
        <div class="info-banner">
        ℹ️ Place trained <code>.pkl</code> / <code>.pth</code> files in the same folder to enable ML models.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#555;text-align:center;'>"
        "Based on Bieniawski (1989) · Lowson & Bieniawski (2013)<br>"
        "Rehman et al. (2018) · Hoek & Marinos (2000)"
        "</div>", unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
X_raw = [ucs, rqd, js, jc, gw, orient_adj, span, depth, method]

# Always compute physics baseline
phys = physics_predict(ucs, rqd, js, jc, gw, orient_adj, span, depth, method)

# Select display result
if selected_model == "Physics Engine (Bieniawski 1989)":
    result = phys
elif selected_model == "Ridge / Lasso + LogReg":
    result = ml_predict(models, X_raw, 'lr') or phys
elif selected_model == "SVM / SVR":
    result = ml_predict(models, X_raw, 'svm') or phys
elif selected_model == "Random Forest":
    result = ml_predict(models, X_raw, 'rf') or phys
elif selected_model == "ANN (PyTorch)":
    result = ml_predict(models, X_raw, 'ann') or phys
else:
    result = phys

rmr_val      = result['rmr']
rock_class   = result['class']
bolt_density = result['bolt_density']
bolt_length  = result['bolt_length']
shotcrete    = result['shotcrete']

ratings = get_individual_ratings(ucs, rqd, js, jc, gw, orient_adj)
cls_info = CLASS_DESC.get(rock_class, CLASS_DESC['III'])

# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# ⛏️ RMR Tunnel Support Designer")
st.markdown(
    "<div style='color:#8892b0;font-size:0.9rem;margin-top:-8px;margin-bottom:16px;'>"
    "Rock Mass Rating (RMR89) — Bieniawski 1989 | Real-time tunnel support design"
    "</div>", unsafe_allow_html=True
)

# ── TOP ROW — key results ──────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    bar_color = CLASS_COLORS.get(rock_class, '#aaa')
    st.markdown(f"""
    <div class="metric-card">
      <div class="label">RMR Score</div>
      <div class="value" style="color:{bar_color}">{rmr_val:.1f}</div>
      <div class="unit">out of 100</div>
      <div class="rmr-bar-outer" style="margin-top:8px;">
        <div class="rmr-bar-inner" style="width:{min(rmr_val,100)}%;background:{bar_color};"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    label, color, _ = cls_info
    st.markdown(f"""
    <div class="metric-card">
      <div class="label">Rock Class</div>
      <div style="margin:8px 0;">
        <span class="class-badge" style="background:{color}22;color:{color};border:2px solid {color};">
          Class {rock_class}
        </span>
      </div>
      <div class="unit">{label}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="label">Bolt Density</div>
      <div class="value">{bolt_density:.2f}</div>
      <div class="unit">bolts / m²</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
      <div class="label">Bolt Length</div>
      <div class="value">{bolt_length:.2f}</div>
      <div class="unit">metres</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
      <div class="label">Shotcrete</div>
      <div class="value">{shotcrete:.0f}</div>
      <div class="unit">mm (crown)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── SECOND ROW — RMR breakdown + recommendations ──────────────────────────
left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown("### 📊 RMR89 Rating Breakdown")
    total = sum(v for v in ratings.values())
    max_vals = {
        'R1 UCS': 15, 'R2 RQD': 20, 'R3 Joint Spacing': 20,
        'R4 Joint Cond.': 30, 'R5 Groundwater': 15, 'B  Orientation': 0
    }

    for param, val in ratings.items():
        max_v = max_vals.get(param, 12)
        pct   = (val / max_v * 100) if max_v > 0 else 100
        # colour: green if high fraction, red if low
        r_frac = val / max_v if max_v > 0 else 0
        bar_c  = f"hsl({int(r_frac * 120)}, 70%, 45%)"
        sign   = "+" if val >= 0 else ""
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin:5px 0;">
          <span style="width:170px;font-size:0.8rem;color:#c9d1d9;font-family:monospace;">{param}</span>
          <div style="flex:1;background:#21262d;border-radius:4px;height:14px;overflow:hidden;">
            <div style="width:{max(0,pct):.0f}%;background:{bar_c};height:100%;border-radius:4px;"></div>
          </div>
          <span style="width:60px;text-align:right;font-size:0.82rem;color:#e6edf3;font-weight:600;">{sign}{val} / {max_v if max_v>0 else '—'}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:10px;padding:8px 12px;background:#1a2a1a;border-radius:8px;
                border:1px solid #2ecc7144;display:flex;justify-content:space-between;">
      <span style="color:#8892b0;font-size:0.85rem;">Total RMR = Σ Ratings</span>
      <span style="color:#2ecc71;font-weight:700;font-size:1.05rem;">{total} / 100</span>
    </div>
    """, unsafe_allow_html=True)

    # Depth stress note
    df = depth_stress_factor(phys['rmr'], depth, ucs)
    if df > 1.0:
        st.markdown(f"""
        <div class="warn-banner" style="margin-top:10px;">
        ⚠️ <b>Depth stress modifier active:</b> σ<sub>v</sub>/σ<sub>cm</sub> ratio exceeds squeezing
        threshold (Hoek & Marinos 2000). Support increased by <b>×{df:.2f}</b>.
        {"Bieniawski guidelines exceed σ<sub>v</sub>&gt;25 MPa validity." if depth > 926 else ""}
        </div>
        """, unsafe_allow_html=True)

with right:
    st.markdown("### 🔩 Support Recommendation")

    label, color, desc = cls_info
    method_str = "TBM" if method == 0 else "Drill & Blast"

    st.markdown(f"""
    <div style="background:{color}18;border:1.5px solid {color};border-radius:12px;padding:1rem 1.2rem;margin-bottom:12px;">
      <div style="font-size:1.1rem;font-weight:700;color:{color};margin-bottom:6px;">
        Class {rock_class} — {label}
      </div>
      <div style="font-size:0.84rem;color:#c9d1d9;">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="formula-box">
    Bolt Length  = max(base, α×span) × method_factor × depth_factor<br>
    &nbsp;&nbsp;= max({base_bolt_length(phys['rmr']):.1f}m, {span_fraction(phys['rmr']):.2f}×{span:.1f}m) {"× 0.90 (TBM)" if method==0 else ""} × {depth_stress_factor(phys['rmr'],depth,ucs):.2f}<br>
    &nbsp;&nbsp;= <b>{bolt_length:.2f} m</b><br><br>
    Shotcrete    = base({phys['class']}) × (span/10) {"× 0.80 (TBM)" if method==0 else ""} × df<br>
    &nbsp;&nbsp;= {base_shotcrete_mm(phys['rmr']):.0f} × {span/10:.2f} {"× 0.80" if method==0 else ""} × {depth_stress_factor(phys['rmr'],depth,ucs):.2f}<br>
    &nbsp;&nbsp;= <b>{shotcrete:.0f} mm</b>
    </div>
    """, unsafe_allow_html=True)

    # Excavation method note
    if method == 0:
        st.markdown("""
        <div class="info-banner">
        ℹ️ <b>TBM reductions applied:</b> Bolt length ×0.90, Shotcrete ×0.80
        (Barton et al. 1974; Rehman et al. 2018)
        </div>
        """, unsafe_allow_html=True)

# ── THIRD ROW — comparison table + reference ─────────────────────────────
st.markdown("---")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### 📋 Model Comparison")

    if selected_model != "Physics Engine (Bieniawski 1989)":
        phys_r = phys
        ml_r   = result
        comp_rows = [
            ("RMR Score",      f"{phys_r['rmr']:.1f}",           f"{ml_r['rmr']:.1f}"),
            ("Rock Class",     f"Class {phys_r['class']}",        f"Class {ml_r['class']}"),
            ("Bolt Density",   f"{phys_r['bolt_density']:.2f} /m²", f"{ml_r['bolt_density']:.2f} /m²"),
            ("Bolt Length",    f"{phys_r['bolt_length']:.2f} m",  f"{ml_r['bolt_length']:.2f} m"),
            ("Shotcrete",      f"{phys_r['shotcrete']:.0f} mm",   f"{ml_r['shotcrete']:.0f} mm"),
        ]
        tbl = "<table class='ref-table'><tr><th>Target</th><th>Physics</th><th>" + selected_model + "</th></tr>"
        for row in comp_rows:
            tbl += f"<tr><td>{row[0]}</td><td>{row[1]}</td><td><b>{row[2]}</b></td></tr>"
        tbl += "</table>"
        st.markdown(tbl, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-banner">
        Select an ML model from the sidebar to compare against the physics engine.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("#### Bieniawski (1989) Support Guidelines")
        st.markdown(SUPPORT_TABLE_HTML, unsafe_allow_html=True)

with col_b:
    st.markdown("### 🗒️ All Classes — RMR Reference")
    st.markdown(SUPPORT_TABLE_HTML, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📐 Key Equations")
    st.markdown("""
    <div class="formula-box">
    RMR89 = R1(UCS) + R2(RQD) + R3(Js) + R4(Jc) + R5(GW) + B(Orientation)<br><br>
    σ_v = 0.027 × H (MPa)  [vertical stress]<br>
    σ_cm = 0.5 × UCS × exp((RMR-100)/24)  [rock mass strength]<br>
    df   = min(1.5, 1 + 0.3 × max(0, σ_v/σ_cm − 0.2))  [depth factor]
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="font-size:0.74rem;color:#555;text-align:center;line-height:1.6;">
Bieniawski, Z.T. (1989). <em>Engineering Rock Mechanics.</em> Balkema. &nbsp;|&nbsp;
Lowson & Bieniawski (2013). RETC Proc. &nbsp;|&nbsp;
Rehman et al. (2018). <em>Applied Sciences, 8(5), 782.</em> &nbsp;|&nbsp;
Hoek & Marinos (2000). <em>Tunnels & Tunnelling Int.</em> &nbsp;|&nbsp;
Barton et al. (1974). <em>Rock Mechanics, 6(4).</em>
</div>
""", unsafe_allow_html=True)
