"""
Generate presentation-ready statistic graphics for GNSS Guardian.

Run from project root:
    python3 presentation_stats.py

Saves five PNGs into ./presentation/, each ~1600px wide, large fonts,
high contrast - designed for slides / poster.
"""
from __future__ import annotations
import pathlib, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

sys.path.append(str(pathlib.Path(__file__).parent))
from gps_detection_utils import train_rf

OUT = pathlib.Path(__file__).parent / 'presentation'
OUT.mkdir(exist_ok=True)

# --- Big, slide-friendly defaults ---
plt.rcParams.update({
    'figure.dpi': 140,
    'font.size': 13,
    'axes.titlesize': 16,
    'axes.labelsize': 13,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
})

GREEN = '#4caf50'
RED   = '#e53935'
BLUE  = '#1976d2'
GRAY  = '#9e9e9e'

# ---------------------------------------------------------------------------
# 1. Load both datasets (real if present, simulated otherwise) + train
# ---------------------------------------------------------------------------
def load_uav():
    sp_p = '../UAVAttackData/Live GPS Spoofing and Jamming/GPS Spoofing/Processed/spoofing-merged-gps-only.csv'
    jm_p = '../UAVAttackData/Live GPS Spoofing and Jamming/GPS Jamming/Processed/jamming-merged-gps-only.csv'
    try:
        sp = pd.read_csv(sp_p); sp['recording'] = 'spoofing_csv'
        jm = pd.read_csv(jm_p); jm['recording'] = 'jamming_csv'
        df = pd.concat([sp, jm], ignore_index=True)
        df['is_attack'] = (df['label'] == 'malicious').astype(int)
        return df, 'real'
    except FileNotFoundError:
        return None, None

def load_av():
    p = '../AV-GPS-Dataset/AV-GPS-Dataset-1.csv'
    try:
        df = pd.read_csv(p); df['recording'] = 'real_drive'
        return df, 'real'
    except FileNotFoundError:
        return None, None

print("Loading data + training models …")
uav, uav_src = load_uav()
av,  av_src  = load_av()

# Feature engineering and rules below mirror the notebooks 1:1, including the
# fact that UAV lat/lon stays in 1e-7 deg units inside step_metres - that's
# how the notebook produces the 41% rules baseline reported in the submission.
if uav is not None:
    g = uav.groupby('recording', sort=False)
    uav['speed_3d']    = np.linalg.norm(uav[['vel_n_m_s', 'vel_e_m_s', 'vel_d_m_s']], axis=1)
    uav['lat_delta']   = g['lat_y'].diff().abs().fillna(0)
    uav['lon_delta']   = g['lon_y'].diff().abs().fillna(0)
    uav['alt_delta']   = g['alt_y'].diff().abs().fillna(0)
    uav['step_metres'] = np.hypot(uav['lat_delta'], uav['lon_delta']) * 111_320
    uav['gps_quality'] = uav['hdop'] + uav['vdop']
    uav_FEATS = ['speed_3d', 'step_metres', 'alt_delta', 'eph_y', 'epv_y',
                 'hdop', 'vdop', 'gps_quality', 'satellites_used', 'jamming_indicator']

    # Submission rules: speed > 55 m/s, step > 5 m, gps_quality > 6
    uav_rule_alarm = ((uav['speed_3d'] > 55) | (uav['step_metres'] > 5) | (uav['gps_quality'] > 6)).astype(int)
    uav_rule_acc   = (uav_rule_alarm == uav['is_attack']).mean()

    uav_model, uav_Xte, uav_yte = train_rf(uav, uav_FEATS, 'is_attack')
    uav_yp = uav_model.predict(uav_Xte)

if av is not None:
    g = av.groupby('recording', sort=False)
    av['speed_kmh']   = av['Velocity (m/s)'] * 3.6
    av['lat_delta']   = g['GPS Latitude'].diff().abs().fillna(0)
    av['lon_delta']   = g['GPS Longitude'].diff().abs().fillna(0)
    av['step_metres'] = np.hypot(av['lat_delta'], av['lon_delta']) * 111_320
    av['acceleration'] = av['Velocity (m/s)'].diff().abs().fillna(0)
    av['heading_delta'] = g['GPS Course'].diff().abs().fillna(0)
    av['heading_delta'] = np.where(av['heading_delta'] > 180,
                                   360 - av['heading_delta'], av['heading_delta'])
    av['gps_quality'] = av['GPS HDOP'] + av['GPS VDOP']
    av_FEATS = ['Velocity (m/s)', 'speed_kmh', 'acceleration',
                'step_metres', 'heading_delta',
                'GPS HDOP', 'GPS VDOP', 'gps_quality',
                'Satellite Count', 'Satellite Locks']

    # Submission rules: speed > 300 km/h, step > 50 m, gps_quality > 6, |Δheading| > 90°
    av_rule_alarm = ((av['speed_kmh']     > 300) |
                     (av['step_metres']   > 50)  |
                     (av['gps_quality']   > 6)   |
                     (av['heading_delta'] > 90)).astype(int)
    av_rule_acc   = (av_rule_alarm == av['Data Type']).mean()

    av_model, av_Xte, av_yte = train_rf(av, av_FEATS, 'Data Type')
    av_yp = av_model.predict(av_Xte)


# ---------------------------------------------------------------------------
# Slide 1: headline accuracy bar chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5.5))
labels  = ['UAV\n(drone)', 'AV\n(vehicle)']
ml_acc  = [accuracy_score(uav_yte, uav_yp) if uav is not None else 0,
           accuracy_score(av_yte,  av_yp)  if av  is not None else 0]
rule_acc = [uav_rule_acc if uav is not None else 0,
            av_rule_acc  if av  is not None else 0]
x = np.arange(len(labels))
w = 0.35
b1 = ax.bar(x - w/2, ml_acc,  w, label='RandomForest', color=BLUE)
b2 = ax.bar(x + w/2, rule_acc, w, label='Physics rules', color=GRAY)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f'{b.get_height():.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=14)
ax.set_ylim(0, 1.08)
ax.set_ylabel('Accuracy on held-out test set')
ax.set_title('GNSS Guardian - detection accuracy across domains', pad=12)
ax.legend(loc='lower right', fontsize=12)
plt.tight_layout()
plt.savefig(OUT / '01_accuracy_comparison.png', dpi=160, bbox_inches='tight')
plt.close()
print(f"  saved {OUT / '01_accuracy_comparison.png'}")


# ---------------------------------------------------------------------------
# Slide 2: attack signature - feature ratio attack/benign
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 5))

if uav is not None:
    pairs = [('speed_3d', 'speed (m/s)'),
             ('step_metres', 'GPS jitter'),
             ('eph_y', 'eph_y'),
             ('hdop', 'HDOP')]
    ratios = []; labels = []
    for col, lab in pairs:
        b = uav.loc[uav['is_attack'] == 0, col].mean()
        a = uav.loc[uav['is_attack'] == 1, col].mean()
        if b > 0:
            ratios.append(a / b); labels.append(lab)
    bars = ax[0].barh(labels, ratios, color=RED, alpha=0.85)
    ax[0].axvline(1, color='black', lw=1, ls='--')
    for bar, r in zip(bars, ratios):
        ax[0].text(r + 0.1, bar.get_y() + bar.get_height()/2, f'{r:.1f}×',
                   va='center', fontsize=12, fontweight='bold')
    ax[0].set_title('UAV - what changes during attack')
    ax[0].set_xlabel('attack mean / benign mean')

if av is not None:
    pairs = [('speed_kmh', 'speed (km/h)'),
             ('GPS HDOP', 'HDOP'),
             ('GPS VDOP', 'VDOP'),
             ('Satellite Count', 'sat count')]
    ratios = []; labels = []
    for col, lab in pairs:
        b = av.loc[av['Data Type'] == 0, col].mean()
        a = av.loc[av['Data Type'] == 1, col].mean()
        if b > 0:
            ratios.append(a / b); labels.append(lab)
    colors = [RED if r > 1 else BLUE for r in ratios]
    bars = ax[1].barh(labels, ratios, color=colors, alpha=0.85)
    ax[1].axvline(1, color='black', lw=1, ls='--')
    for bar, r in zip(bars, ratios):
        ax[1].text(r + 0.5, bar.get_y() + bar.get_height()/2, f'{r:.1f}×',
                   va='center', fontsize=12, fontweight='bold')
    ax[1].set_title('AV - what changes during attack')
    ax[1].set_xlabel('attack mean / benign mean')
    ax[1].set_xlim(0, max(ratios) * 1.15)

fig.suptitle('Attack signatures: how each feature shifts when GPS is spoofed', y=1.02, fontsize=15)
plt.tight_layout()
plt.savefig(OUT / '02_attack_signature.png', dpi=160, bbox_inches='tight')
plt.close()
print(f"  saved {OUT / '02_attack_signature.png'}")


# ---------------------------------------------------------------------------
# Slide 3: confusion matrices side by side
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for axx, y_te, y_pred, title in [
    (ax[0], uav_yte if uav is not None else None, uav_yp if uav is not None else None, 'UAV - RandomForest'),
    (ax[1], av_yte  if av  is not None else None, av_yp  if av  is not None else None, 'AV - RandomForest'),
]:
    if y_te is None: continue
    cm = confusion_matrix(y_te, y_pred)
    cmn = cm / cm.sum(axis=1, keepdims=True) * 100  # row-normalised %
    im = axx.imshow(cmn, cmap='Blues', vmin=0, vmax=100)
    axx.set_xticks([0, 1], ['benign', 'attack'])
    axx.set_yticks([0, 1], ['benign', 'attack'])
    axx.set_xlabel('Predicted'); axx.set_ylabel('Actual')
    axx.set_title(title)
    axx.grid(False)
    for i in range(2):
        for j in range(2):
            txt = f'{cmn[i,j]:.1f}%\n({cm[i,j]:,})'
            color = 'white' if cmn[i, j] > 50 else 'black'
            axx.text(j, i, txt, ha='center', va='center', fontsize=12, fontweight='bold', color=color)
fig.suptitle('Confusion matrices - row-normalised', y=1.02, fontsize=15)
plt.tight_layout()
plt.savefig(OUT / '03_confusion_matrices.png', dpi=160, bbox_inches='tight')
plt.close()
print(f"  saved {OUT / '03_confusion_matrices.png'}")


# ---------------------------------------------------------------------------
# Slide 4: dataset summary card
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5))
ax.axis('off')

rows = []
if uav is not None:
    rows.append(['UAV (IEEE DataPort)', f"{len(uav):,}",
                 f"{uav['is_attack'].mean():.1%}",
                 'spoofing + jamming',
                 'real HackRF lab attack'])
if av is not None:
    rows.append(['AV (Univ. of Arizona)', f"{len(av):,}",
                 f"{av['Data Type'].mean():.1%}",
                 'spoofing',
                 'ACL-Rover testbed'])

cols = ['Dataset', 'Samples', 'Attack rate', 'Attack type', 'Source']
table = ax.table(cellText=rows, colLabels=cols, cellLoc='center', loc='center',
                 colWidths=[0.25, 0.13, 0.15, 0.20, 0.22])
table.auto_set_font_size(False)
table.set_fontsize(13)
table.scale(1, 2.4)
for j in range(len(cols)):
    cell = table[(0, j)]
    cell.set_facecolor('#1976d2'); cell.set_text_props(color='white', fontweight='bold')
ax.set_title('Datasets used in evaluation', pad=20, fontsize=16)
plt.tight_layout()
plt.savefig(OUT / '04_datasets_summary.png', dpi=160, bbox_inches='tight')
plt.close()
print(f"  saved {OUT / '04_datasets_summary.png'}")


# ---------------------------------------------------------------------------
# Slide 5: full-metric matrix (precision, recall, F1, AUC for both datasets)
# ---------------------------------------------------------------------------
def metrics(y_te, y_pred, y_proba):
    return {
        'Accuracy':  accuracy_score(y_te, y_pred),
        'Precision': precision_score(y_te, y_pred, zero_division=0),
        'Recall':    recall_score(y_te, y_pred, zero_division=0),
        'F1':        f1_score(y_te, y_pred, zero_division=0),
        'ROC AUC':   roc_auc_score(y_te, y_proba),
    }

stat_rows = []
if uav is not None:
    stat_rows.append(('UAV', metrics(uav_yte, uav_yp, uav_model.predict_proba(uav_Xte)[:, 1])))
if av is not None:
    stat_rows.append(('AV',  metrics(av_yte,  av_yp,  av_model.predict_proba(av_Xte)[:, 1])))

if stat_rows:
    fig, ax = plt.subplots(figsize=(11, 5))
    metric_names = list(stat_rows[0][1].keys())
    x = np.arange(len(metric_names))
    w = 0.35
    for i, (lab, m) in enumerate(stat_rows):
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, list(m.values()), w, label=lab,
                      color=BLUE if lab == 'UAV' else GREEN)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    f'{b.get_height():.3f}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.08)
    ax.set_title('All evaluation metrics - UAV vs AV', pad=12)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / '05_all_metrics.png', dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  saved {OUT / '05_all_metrics.png'}")

# ---------------------------------------------------------------------------
# Slide 6: per-class precision / recall / F1 (the metrics that matter when
# accuracy alone is misleading on imbalanced data)
# ---------------------------------------------------------------------------
def per_class_prf(y_te, y_pred):
    """Return {'benign': (P,R,F), 'attack': (P,R,F)}."""
    out = {}
    for cls, name in [(0, 'benign'), (1, 'attack')]:
        # treat current class as positive, the other as negative
        yt = (np.asarray(y_te)    == cls).astype(int)
        yp = (np.asarray(y_pred)  == cls).astype(int)
        out[name] = (precision_score(yt, yp, zero_division=0),
                     recall_score   (yt, yp, zero_division=0),
                     f1_score       (yt, yp, zero_division=0))
    return out

panels = []
if uav is not None: panels.append(('UAV', uav_yte, uav_yp))
if av  is not None: panels.append(('AV',  av_yte,  av_yp))

if panels:
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5.5), sharey=True)
    if len(panels) == 1:
        axes = [axes]

    metric_labels = ['Precision', 'Recall', 'F1']
    x = np.arange(len(metric_labels))
    w = 0.36

    for ax, (dom, y_te, y_pred) in zip(axes, panels):
        prf = per_class_prf(y_te, y_pred)
        b_vals = prf['benign']
        a_vals = prf['attack']

        b1 = ax.bar(x - w/2, b_vals, w, color=GREEN, label='benign')
        b2 = ax.bar(x + w/2, a_vals, w, color=RED,   label='attack')
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.012,
                        f'{b.get_height():.3f}', ha='center', va='bottom',
                        fontsize=11, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(metric_labels, fontsize=13)
        ax.set_ylim(0, 1.08)
        ax.set_title(f'{dom} - per-class metrics')
        ax.legend(loc='lower right', fontsize=11)
    axes[0].set_ylabel('Score (0 - 1)')
    fig.suptitle('Precision / Recall / F1 broken down by class', y=1.02, fontsize=15)
    plt.tight_layout()
    plt.savefig(OUT / '06_precision_recall_f1.png', dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  saved {OUT / '06_precision_recall_f1.png'}")


# ---------------------------------------------------------------------------
# Slide 7: precision / recall / F1 vs threshold sweep - shows what happens
# if the operator wants more sensitivity (or fewer false alarms)
# ---------------------------------------------------------------------------
from sklearn.metrics import precision_recall_curve

if panels:
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5.5), sharey=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (dom, y_te, _) in zip(axes, panels):
        model = uav_model if dom == 'UAV' else av_model
        Xte   = uav_Xte   if dom == 'UAV' else av_Xte
        proba = model.predict_proba(Xte)[:, 1]

        thr = np.linspace(0.01, 0.99, 200)
        prec, rec, f1s = [], [], []
        for t in thr:
            yp = (proba >= t).astype(int)
            prec.append(precision_score(y_te, yp, zero_division=0))
            rec.append(recall_score   (y_te, yp, zero_division=0))
            f1s.append(f1_score       (y_te, yp, zero_division=0))

        ax.plot(thr, prec, label='Precision', color=BLUE,  lw=2.2)
        ax.plot(thr, rec,  label='Recall',    color=RED,   lw=2.2)
        ax.plot(thr, f1s,  label='F1',        color='#7b1fa2', lw=2.2)
        ax.axvline(0.5, color='gray', ls='--', alpha=0.6, label='default 0.5')
        # mark the F1 maximum
        i_best = int(np.argmax(f1s))
        ax.scatter([thr[i_best]], [f1s[i_best]], s=80, color='#7b1fa2',
                   edgecolor='white', zorder=5)
        ax.annotate(f'best F1 = {f1s[i_best]:.3f}\nthr = {thr[i_best]:.2f}',
                    xy=(thr[i_best], f1s[i_best]),
                    xytext=(thr[i_best] + 0.05, max(0.55, f1s[i_best] - 0.18)),
                    fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#7b1fa2', lw=1.2))

        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.set_xlabel('Decision threshold (attack class probability ≥ t)')
        ax.set_title(f'{dom} - sweep over decision threshold')
        ax.legend(loc='lower left', fontsize=11)
    axes[0].set_ylabel('Score (0 - 1)')
    fig.suptitle('How precision, recall and F1 trade off as the alarm threshold moves',
                 y=1.02, fontsize=15)
    plt.tight_layout()
    plt.savefig(OUT / '07_threshold_sweep.png', dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  saved {OUT / '07_threshold_sweep.png'}")


print(f"\nDone. {len(list(OUT.glob('*.png')))} PNGs in {OUT.relative_to(pathlib.Path.cwd().parent)}/")
