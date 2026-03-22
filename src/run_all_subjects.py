# =============================================================================
# run_all_subjects.py
# Multi-subject pipeline: Load → Preprocess → Feature Extraction → Cache → Train
# =============================================================================

import os
import numpy as np
import pandas as pd
import mne
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, classification_report

# Import từ file có sẵn — KHÔNG copy code
from Flex2023_moabb import Flex2023_moabb
from eeg_pipeline import (
    merge_epochs_labels,
    build_feature_matrix,
    MVAR_ORDER, ALPHA_BAND, TOP_PERCENT,
    CLASS_LABELS, MERGE_LEGS
)


# =============================================================================
# CONFIG — chỉnh ở đây
# =============================================================================

OUTPUT_DIR  = "/Users/audaodoantruc/Downloads/IU-MIBCI_4class/output"
CACHE_DIR   = os.path.join(OUTPUT_DIR, "cache/")

# Theo timeline thực tế: marker t=0, chuông t=4, MI kết thúc t=8
EPOCH_TMIN  = 3.5   # 0.5s trước tiếng chuông (làm baseline)
EPOCH_TMAX  = 8.0   # hết 4s MI

# Baseline = 0.5s trước tiếng chuông
BASELINE    = (3.5, 4.0)

# MI event IDs (sau khi fix_stim: 1,2,3,4)
MI_EVENT_ID = {'Tay phải': 1, 'Tay trái': 2, 'Chân phải': 3, 'Chân trái': 4}

# Tạo thư mục nếu chưa có
os.makedirs(CACHE_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# SECTION 1 — XỬ LÝ 1 SUBJECT
# =============================================================================

def process_one_subject(raw, subject_id):
    """
    Nhận MNE Raw object (32 EEG + 1 Stim) từ Flex2023_moabb.
    Trả về (X, y) — feature matrix và nhãn.
    """
    try:
        fs = int(raw.info['sfreq'])

        # ── 1. Tách EEG và Stim ──────────────────────────────────────────────
        eeg_chs   = [ch for ch in raw.ch_names if ch != 'Stim']
        eeg_data  = raw.get_data(picks=eeg_chs)
        stim_data = raw.get_data(picks=['Stim'])

        # ── 2. Bandpass 8–30 Hz + Notch 50 Hz ────────────────────────────────
        info_eeg = mne.create_info(ch_names=eeg_chs,
                                   ch_types=['eeg'] * len(eeg_chs),
                                   sfreq=fs)
        raw_eeg = mne.io.RawArray(eeg_data, info_eeg, verbose=False)
        raw_eeg.filter(l_freq=8, h_freq=30, verbose=False)
        raw_eeg.notch_filter(freqs=50, verbose=False)

        # ── 3. Ghép lại EEG đã lọc + Stim ────────────────────────────────────
        ch_names_full = eeg_chs + ['Stim']
        ch_types_full = ['eeg'] * len(eeg_chs) + ['stim']
        info_full = mne.create_info(ch_names=ch_names_full,
                                    ch_types=ch_types_full,
                                    sfreq=fs)
        data_full = np.vstack([raw_eeg.get_data(), stim_data])
        raw_full  = mne.io.RawArray(data_full, info_full, verbose=False)

        # ── 4. Find events từ kênh Stim ───────────────────────────────────────
        events = mne.find_events(raw_full, stim_channel='Stim', verbose=False)

        # Lọc chỉ giữ MI events (1, 2, 3, 4) — bỏ biocalib (20+), kines (10+)
        mi_mask = np.isin(events[:, 2], list(MI_EVENT_ID.values()))
        events  = events[mi_mask]
        print(f"  Sub F{subject_id}: {len(events)} MI trials")

        if len(events) == 0:
            print(f"  ⚠️  Không có MI event — bỏ qua subject này")
            return None, None

        # ── 5. Epoching ───────────────────────────────────────────────────────
        epochs = mne.Epochs(
            raw_full, events,
            event_id = MI_EVENT_ID,
            tmin     = EPOCH_TMIN,
            tmax     = EPOCH_TMAX,
            baseline = BASELINE,
            preload  = True,
            verbose  = False
        )
        print(f"  Epochs sau khi drop: {len(epochs)}")

        # ── 6. Gộp Chân phải + Chân trái → "Chân" ────────────────────────────
        epochs = merge_epochs_labels(epochs, MERGE_LEGS)

        # ── 7. Feature extraction ─────────────────────────────────────────────
        X, y = build_feature_matrix(
            epochs, CLASS_LABELS,
            order       = MVAR_ORDER,
            fs          = fs,
            top_percent = TOP_PERCENT
        )
        return X, y

    except Exception as e:
        print(f"  ❌ Subject F{subject_id} lỗi: {e}")
        return None, None


# =============================================================================
# SECTION 2 — LOAD VÀ CACHE
# =============================================================================

def load_subject_with_cache(dataset, subject_id, force_recompute=False):
    """
    Load feature (X, y) của 1 subject.
    - Nếu đã có cache → đọc thẳng (nhanh).
    - Chưa có cache   → tính rồi lưu lại.
    - force_recompute=True → bỏ qua cache, tính lại từ đầu.
    """
    cache_path = os.path.join(
        CACHE_DIR,
        f"cache_sub{subject_id}"
        f"_order{MVAR_ORDER}"
        f"_alpha{ALPHA_BAND[0]}-{ALPHA_BAND[1]}"
        f"_top{int(TOP_PERCENT*100)}.npz"
    )
    # Tên file cache kèm config → tránh dùng nhầm cache cũ khi đổi tham số
    # Ví dụ: cache_sub10_order5_alpha8-13_top20.npz

    if os.path.exists(cache_path) and not force_recompute:
        print(f"Sub F{subject_id}: ⚡ load cache")
        d    = np.load(cache_path, allow_pickle=True)
        X, y = d['X'], d['y']
        return X, y

    # Chưa có cache → load raw rồi tính
    print(f"Sub F{subject_id}: ⏳ đang tính feature...")
    try:
        sub_data = dataset._get_single_subject_data(subject_id)
        raw      = sub_data["0"]["0"]
    except Exception as e:
        print(f"  ❌ Không load được data subject F{subject_id}: {e}")
        return None, None

    X, y = process_one_subject(raw, subject_id)

    if X is not None and len(X) > 0:
        np.savez(cache_path, X=X, y=y)
        print(f"  ✅ Đã lưu cache: {os.path.basename(cache_path)}")

    return X, y


# =============================================================================
# SECTION 3 — CHẠY TOÀN BỘ SUBJECT
# =============================================================================

def run_all_subjects(subject_list=None, force_recompute=False):
    """
    Chạy toàn bộ subject, trả về X_all, y_all, sub_all.
    subject_list=None → dùng toàn bộ subject trong dataset.
    force_recompute=True → tính lại dù đã có cache.
    """
    dataset = Flex2023_moabb()

    if subject_list is None:
        subject_list = dataset.subject_list

    all_X, all_y, all_sub = [], [], []

    for sub_id in subject_list:
        print(f"\n{'='*55}")
        print(f"  Subject F{sub_id}")
        print(f"{'='*55}")

        X, y = load_subject_with_cache(dataset, sub_id, force_recompute)

        if X is not None and len(X) > 0:
            all_X.append(X)
            all_y.append(y)
            all_sub.extend([sub_id] * len(y))
            print(f"  → X: {X.shape} | classes: {dict(zip(*np.unique(y, return_counts=True)))}")
        else:
            print(f"  ⚠️  Bỏ qua subject F{sub_id}")

    if len(all_X) == 0:
        print("❌ Không có subject nào thành công!")
        return None, None, None

    X_all   = np.concatenate(all_X,  axis=0)
    y_all   = np.concatenate(all_y,  axis=0)
    sub_all = np.array(all_sub)

    print(f"\n{'='*55}")
    print(f"✅ TỔNG KẾT LOAD DATA")
    print(f"   Số subject thành công : {len(np.unique(sub_all))}")
    print(f"   Tổng số epochs        : {X_all.shape[0]}")
    print(f"   Số features           : {X_all.shape[1]}")
    print(f"   Phân bố class         : {dict(zip(*np.unique(y_all, return_counts=True)))}")

    return X_all, y_all, sub_all


# =============================================================================
# SECTION 4 — LƯU FEATURE MATRIX
# =============================================================================

def save_feature_csv(X_all, y_all, sub_all):
    """Lưu toàn bộ feature matrix ra CSV."""
    from eeg_pipeline import EEG_CHANNELS
    ch = EEG_CHANNELS
    feature_names = [f"feature_{i}" for i in range(X_all.shape[1])]
    feat_df = pd.DataFrame(X_all, columns=feature_names)
    feat_df.insert(0, "Subject", sub_all)
    feat_df.insert(1, "Label",   y_all)

    out_path = os.path.join(OUTPUT_DIR, "features_all_subjects.csv")
    feat_df.to_csv(out_path, index=False)
    print(f"\n✅ Đã lưu feature CSV → {out_path}")
    return feat_df


# =============================================================================
# SECTION 5 — TRAIN VÀ ĐÁNH GIÁ (LOSO)
# =============================================================================

def train_loso(X_all, y_all, sub_all):
    """
    Leave-One-Subject-Out Cross-Validation với SVM.
    Mỗi lần: train trên N-1 subject, test trên 1 subject còn lại.
    """
    print(f"\n{'='*55}")
    print("  LOSO CROSS-VALIDATION")
    print(f"{'='*55}")

    # Encode nhãn string → số
    le      = LabelEncoder()
    y_enc   = le.fit_transform(y_all)

    logo    = LeaveOneGroupOut()
    scaler  = StandardScaler()
    clf     = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)

    scores, results = [], []

    for train_idx, test_idx in logo.split(X_all, y_enc, groups=sub_all):
        sub_test = np.unique(sub_all[test_idx])[0]

        X_tr = scaler.fit_transform(X_all[train_idx])
        X_te = scaler.transform(X_all[test_idx])

        clf.fit(X_tr, y_enc[train_idx])
        y_pred = clf.predict(X_te)

        acc = accuracy_score(y_enc[test_idx], y_pred)
        scores.append(acc)
        results.append({"Subject": f"F{sub_test}", "Accuracy": acc})
        print(f"  Sub F{sub_test:>2}: acc = {acc:.3f}")

    # Tổng kết
    mean_acc = np.mean(scores)
    std_acc  = np.std(scores)
    print(f"\n{'='*55}")
    print(f"  🏆 LOSO Mean Accuracy : {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"  Best  subject         : F{results[np.argmax(scores)]['Subject']} ({max(scores):.3f})")
    print(f"  Worst subject         : F{results[np.argmin(scores)]['Subject']} ({min(scores):.3f})")

    # Lưu kết quả ra CSV
    results_df = pd.DataFrame(results)
    results_df.loc[len(results_df)] = {"Subject": "MEAN ± STD",
                                        "Accuracy": f"{mean_acc:.3f} ± {std_acc:.3f}"}
    out_path = os.path.join(OUTPUT_DIR, "loso_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"  Kết quả LOSO → {out_path}")

    return scores, results_df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # ── Bước 1: Load + tính feature toàn bộ subject ──────────────────────────
    # subject_list=None  → chạy tất cả subject trong dataset
    # force_recompute=False → dùng cache nếu có (nhanh hơn)
    # force_recompute=True  → tính lại từ đầu (khi đổi config)
    X_all, y_all, sub_all = run_all_subjects(
        subject_list   = None,
        force_recompute= False
    )

    if X_all is None:
        exit()

    # ── Bước 2: Lưu feature matrix ra CSV ────────────────────────────────────
    save_feature_csv(X_all, y_all, sub_all)

    # ── Bước 3: Train và đánh giá LOSO ───────────────────────────────────────
    scores, results_df = train_loso(X_all, y_all, sub_all)
