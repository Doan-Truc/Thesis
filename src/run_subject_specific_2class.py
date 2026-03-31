# =============================================================================
# run_subject_specific.py
# Subject-specific pipeline:
# Load data → Epoch (0-2s) → Balance class → 80/20 split → Feature → Train ML
# 2-class: Tay phải vs Tay trái
# =============================================================================

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import mne
from train_model import MODELS
from Flex2023_moabb import Flex2023_moabb, SUBJECT_GROUP, LIST_SUBJECTS
from eeg_pipeline import (
    estimate_mvar_yw_single,
    compute_dtf,
    apply_threshold,
    extract_features_single_epoch,
    MVAR_ORDER, ALPHA_BAND, TOP_PERCENT
)


# =============================================================================
# CONFIG
# =============================================================================

OUTPUT_DIR  = "/Users/audaodoantruc/Downloads/IU-MIBCI_4class/output/subject_specific/"
CACHE_DIR   = os.path.join(OUTPUT_DIR, "cache/")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)


BASELINE    = None   # không dùng baseline vì đã filter
EPOCH_TMIN = 4.0       # bắt đầu từ tiếng chuông
EPOCH_TMAX = 8.0     # hết 2s MI



# MI event IDs (sau fix_stim) — chỉ giữ tay phải & tay trái
MI_EVENT_ID = {'Tay phải': 1, 'Tay trái': 2}

# 2 class
CLASS_LABELS = ['Tay phải', 'Tay trái']




# =============================================================================
# SECTION 1 — LOAD RAW + EPOCH
# =============================================================================

MOTOR_CHANNELS = [
    'FC5', 'FC1', 'FC2', 'FC6',
    'C3',  'Cz',  'C4',
    'CP5', 'CP1', 'CP2', 'CP6',
]

def load_and_epoch(raw, tmin, tmax):
    fs = int(raw.info['sfreq'])

    eeg_chs   = [ch for ch in raw.ch_names if ch in MOTOR_CHANNELS]
    eeg_data  = raw.get_data(picks=eeg_chs)
    stim_data = raw.get_data(picks=['Stim'])

    info_eeg = mne.create_info(ch_names=eeg_chs,
                               ch_types=['eeg'] * len(eeg_chs), sfreq=fs)
    raw_eeg = mne.io.RawArray(eeg_data, info_eeg, verbose=False)
    raw_eeg.filter(l_freq=8, h_freq=30, verbose=False)
    raw_eeg.notch_filter(freqs=50, verbose=False)

    info_full = mne.create_info(
        ch_names=eeg_chs + ['Stim'],
        ch_types=['eeg'] * len(eeg_chs) + ['stim'],
        sfreq=fs
    )
    raw_full = mne.io.RawArray(
        np.vstack([raw_eeg.get_data(), stim_data]),
        info_full, verbose=False
    )

    events  = mne.find_events(raw_full, stim_channel='Stim', verbose=False)
    mi_mask = np.isin(events[:, 2], list(MI_EVENT_ID.values()))
    events  = events[mi_mask]

    if len(events) == 0:
        return None

    epochs = mne.Epochs(
        raw_full, events,
        event_id = MI_EVENT_ID,
        tmin     = tmin,
        tmax     = tmax,
        baseline = BASELINE,
        preload  = True,
        verbose  = False
    )
    return epochs


# =============================================================================
# SECTION 2 — BALANCE CLASS (2-class)
# =============================================================================

def balance_and_merge(epochs):
    """
    Balance Tay phải và Tay trái theo min của 2 class.
    Trả về (data_array, labels_array, n_per_class)
    """
    data_right = epochs['Tay phải'].get_data()   # (n, ch, t)
    data_left  = epochs['Tay trái'].get_data()

    n_right = len(data_right)
    n_left  = len(data_left)

    # Số epoch mỗi class = min của 2 class
    n_per_class = min(n_right, n_left)

    print(f"  Tay phải: {n_right} | Tay trái: {n_left}")
    print(f"  → Balance: lấy {n_per_class} epoch/class")

    # Random sample
    rng = np.random.RandomState(42)
    idx_right = rng.choice(n_right, n_per_class, replace=False)
    idx_left  = rng.choice(n_left,  n_per_class, replace=False)

    data = np.concatenate([
        data_right[idx_right],
        data_left[idx_left],
    ], axis=0)

    labels = np.array(
        ['Tay phải'] * n_per_class +
        ['Tay trái'] * n_per_class
    )

    return data, labels, n_per_class




# =============================================================================
# SECTION — FEATURE EXTRACTION
# =============================================================================

def extract_features(data, fs=128):
    """
    data: (n_epochs, n_channels, n_times)
    Trả về X: (n_valid_epochs, n_features)
    """
    X_list, valid_idx = [], []
    for i in range(len(data)):
        feat = extract_features_single_epoch(
            data[i], order=MVAR_ORDER, fs=fs, top_percent=TOP_PERCENT
        )
        if feat is not None:
            X_list.append(feat)
            valid_idx.append(i)
        else:
            print(f"    ⚠️  Epoch {i} lỗi MVAR → bỏ qua")

    return np.array(X_list), np.array(valid_idx)


# =============================================================================
# SECTION 5 — TRAIN 1 SUBJECT
# =============================================================================
def train_subject(subject_id, raw):
    """
    Full pipeline cho 1 subject.
    Trả về dict kết quả các model.
    """
    # ── Cache path ────────────────────────────────────────────────────────────
    cache_path = os.path.join(
        CACHE_DIR,
        f"cache_specific_sub{subject_id}"
        f"_tmin{EPOCH_TMIN}_tmax{EPOCH_TMAX}"
        f"_order{MVAR_ORDER}"
        f"_top{int(TOP_PERCENT*100)}"
        f"_2class.npz"
    )

    # ── Đọc cache nếu có ──────────────────────────────────────────────────────
    if os.path.exists(cache_path):
        print(f"  ⚡ Load cache")
        d      = np.load(cache_path, allow_pickle=True)
        X, y   = d['X'], d['y']
        fs     = int(d['fs'])
    else:
        # ── Tính từ đầu ───────────────────────────────────────────────────────
        print(f"\n  ── Load & Epoch ──")
        epochs = load_and_epoch(raw, EPOCH_TMIN, EPOCH_TMAX)
        if epochs is None or len(epochs) == 0:
            print(f"  ❌ Không có epochs")
            return None

        total_trials = len(epochs)
        fs = int(epochs.info['sfreq'])
        print(f"  Tổng MI trials (2 class): {total_trials}")

        print(f"\n  ── Balance Class ──")
        data, y, n_per_class = balance_and_merge(epochs)

        print(f"\n  ── Feature Extraction ──")
        X, valid_idx = extract_features(data, fs=fs)
        y = y[valid_idx]

        if len(X) == 0:
            print(f"  ❌ Không có feature nào hợp lệ")
            return None

        # Lưu cache
        np.savez(cache_path, X=X, y=y, fs=fs)
        print(f"  ✅ Đã lưu cache: {os.path.basename(cache_path)}")

    print(f"  X: {X.shape} | Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # Train từng model
    print(f"\n  ── Train Models ──")
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    results = {}
    for model_name, model in MODELS.items():
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        acc    = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred,
                                        target_names=CLASS_LABELS,
                                        zero_division=0, output_dict=True)
        results[model_name] = {
            "accuracy" : round(acc, 4),
            "macro_f1" : round(report['macro avg']['f1-score'], 4),
            "precision": round(report['macro avg']['precision'], 4),
            "recall"   : round(report['macro avg']['recall'], 4),
            "report"   : report,
        }
        print(f"  {model_name:<20}: acc={acc:.3f} | macro_f1={report['macro avg']['f1-score']:.3f}")

    return results


# =============================================================================
# SECTION 6 — CHẠY TẤT CẢ SUBJECT
# =============================================================================

def run_all_subjects():
    dataset = Flex2023_moabb()
    all_results = {}

    for sub_id in LIST_SUBJECTS:
        print(f"\n{'='*55}")
        print(f"  Subject F{sub_id}")
        print(f"{'='*55}")

        try:
            sub_data = dataset._get_single_subject_data(sub_id)
            raw      = sub_data["0"]["0"]
        except Exception as e:
            print(f"  ❌ Load thất bại: {e}")
            continue

        results = train_subject(sub_id, raw)
        if results is not None:
            all_results[sub_id] = results

    return all_results


# =============================================================================
# SECTION 7 — TỔNG HỢP & LƯU KẾT QUẢ
# =============================================================================

def summarize(all_results):
    """Tổng hợp kết quả tất cả subject × model."""
    rows = []
    for sub_id, results in all_results.items():
        for model_name, metrics in results.items():
            rows.append({
                "Subject"  : f"F{sub_id}",
                "Model"    : model_name,
                "Accuracy" : metrics["accuracy"],
                "Macro_F1" : metrics["macro_f1"],
                "Precision": metrics["precision"],
                "Recall"   : metrics["recall"],
            })

    df = pd.DataFrame(rows)

    # Bảng mean theo model
    summary = df.groupby("Model")[["Accuracy","Macro_F1"]]\
                .agg(["mean","std"])\
                .round(4)
    summary.columns = ["Mean_Acc","Std_Acc","Mean_F1","Std_F1"]
    summary = summary.sort_values("Mean_Acc", ascending=False)
    summary.insert(0, "Rank", range(1, len(summary)+1))

    print(f"\n{'='*55}")
    print("  🏆 BẢNG XẾP HẠNG MODEL (Subject-Specific, 2-class)")
    print(f"{'='*55}")
    print(summary.to_string())

    # Lưu CSV
    detail_path  = os.path.join(OUTPUT_DIR, "subject_specific_detail_2class.csv")
    summary_path = os.path.join(OUTPUT_DIR, "subject_specific_summary_2class.csv")
    df.to_csv(detail_path,  index=False)
    summary.to_csv(summary_path)
    print(f"\n✅ Chi tiết → {detail_path}")
    print(f"✅ Tổng hợp → {summary_path}")

    return df, summary


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    all_results = run_all_subjects()
    if all_results:
        df, summary = summarize(all_results)
