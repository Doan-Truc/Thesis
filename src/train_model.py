# =============================================================================
# train_model.py
# Đọc feature cache → Train nhiều ML model → So sánh kết quả (LOSO)
# =============================================================================

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, classification_report

# ML Models
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier


from Flex2023_moabb import LIST_SUBJECTS


# =============================================================================
# CONFIG — chỉnh đường dẫn nếu cần
# =============================================================================

OUTPUT_DIR  = "/Users/audaodoantruc/Downloads/IU-MIBCI_4class/output"
CACHE_DIR   = os.path.join(OUTPUT_DIR, "cache/")

MVAR_ORDER  = 5
ALPHA_BAND  = (8, 13)
TOP_PERCENT = 0.20


# =============================================================================
# DANH SÁCH MODEL
# =============================================================================

MODELS = {
    # SVM
    "SVM_RBF"    : SVC(kernel='rbf', C=10, gamma='scale',
                       class_weight='balanced', random_state=42),
    "SVM_Linear" : SVC(kernel='linear', C=1,
                       class_weight='balanced', random_state=42),
    "SVM_Poly"   : SVC(kernel='poly', degree=3,
                       class_weight='balanced', random_state=42),

    # KNN — không hỗ trợ class_weight, giữ nguyên
    "KNN_3"  : KNeighborsClassifier(n_neighbors=3),
    "KNN_5"  : KNeighborsClassifier(n_neighbors=5),
    "KNN_10" : KNeighborsClassifier(n_neighbors=10),

    # Tree-based
    "Decision_Tree" : DecisionTreeClassifier(max_depth=10,
                          class_weight='balanced', random_state=42),
    "Extra_Trees"   : ExtraTreesClassifier(n_estimators=200,
                          class_weight='balanced', random_state=42),
    "Random_Forest" : RandomForestClassifier(n_estimators=200,
                          class_weight='balanced', random_state=42),

    # Boosting — không hỗ trợ class_weight, giữ nguyên
    "AdaBoost"      : AdaBoostClassifier(n_estimators=200, random_state=42),
    "Gradient_Boost": GradientBoostingClassifier(n_estimators=200,
                          learning_rate=0.1, max_depth=4, random_state=42),

    # Linear
    "Logistic_Reg"  : LogisticRegression(max_iter=1000,
                          class_weight='balanced', random_state=42),
}


# =============================================================================
# SECTION 1 — LOAD CACHE
# =============================================================================

def load_all_cache(subject_list=None):
    """Đọc toàn bộ cache .npz đã tính sẵn → gộp X_all, y_all, sub_all."""
    if subject_list is None:
        subject_list = LIST_SUBJECTS

    all_X, all_y, all_sub = [], [], []

    print("📂 Loading cache...\n")
    for sub_id in subject_list:
        cache_path = os.path.join(
            CACHE_DIR,
            f"cache_sub{sub_id}"
            f"_order{MVAR_ORDER}"
            f"_alpha{ALPHA_BAND[0]}-{ALPHA_BAND[1]}"
            f"_top{int(TOP_PERCENT * 100)}.npz"
        )

        if not os.path.exists(cache_path):
            print(f"  ⚠️  Không tìm thấy cache F{sub_id} — bỏ qua")
            continue

        d    = np.load(cache_path, allow_pickle=True)
        X, y = d['X'], d['y']
        all_X.append(X)
        all_y.append(y)
        all_sub.extend([sub_id] * len(y))
        print(f"  ✅ F{sub_id}: {X.shape[0]} epochs")

    X_all   = np.concatenate(all_X,  axis=0)
    y_all   = np.concatenate(all_y,  axis=0)
    sub_all = np.array(all_sub)

    print(f"\n📦 Tổng   : {X_all.shape[0]} epochs | {X_all.shape[1]} features")
    print(f"   Classes : {dict(zip(*np.unique(y_all, return_counts=True)))}")
    print(f"   Subjects: {np.unique(sub_all).tolist()}\n")
    return X_all, y_all, sub_all


# =============================================================================
# SECTION 2 — TRAIN 1 MODEL VỚI LOSO
# =============================================================================

def train_one_model(model, model_name, X_all, y_enc, sub_all, class_names, print_report=True):
    """LOSO cross-validation cho 1 model."""
    logo   = LeaveOneGroupOut()
    scaler = StandardScaler()
    scores, rows = [], []
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in logo.split(X_all, y_enc, groups=sub_all):
        sub_test = np.unique(sub_all[test_idx])[0]

        X_tr = scaler.fit_transform(X_all[train_idx])
        X_te = scaler.transform(X_all[test_idx])

        model.fit(X_tr, y_enc[train_idx])
        y_pred = model.predict(X_te)

        acc = accuracy_score(y_enc[test_idx], y_pred)
        scores.append(acc)
        rows.append({"Subject": f"F{sub_test}", model_name: round(acc, 4)})
        y_true_all.extend(y_enc[test_idx])
        y_pred_all.extend(y_pred)

    mean_acc = np.mean(scores)
    std_acc  = np.std(scores)
    print(f"  {model_name:<20}: {mean_acc:.3f} ± {std_acc:.3f}")

    # In classification report cho từng model
    if print_report:
        print(classification_report(y_true_all, y_pred_all,
                                     target_names=class_names,
                                     zero_division=0))

    return scores, rows, mean_acc, std_acc


# =============================================================================
# SECTION 3 — TRAIN TẤT CẢ MODEL
# =============================================================================

def train_all_models(X_all, y_all, sub_all):
    """Train tất cả model, lưu kết quả ra CSV."""

    # Encode nhãn string → số
    le    = LabelEncoder()
    y_enc = le.fit_transform(y_all)
    print(f"Classes: {le.classes_.tolist()}\n")

    print("=" * 55)
    print("  LOSO CROSS-VALIDATION — TẤT CẢ MODEL")
    print("=" * 55)

    summary_rows = []
    detail_dict  = {}

    for model_name, model in MODELS.items():
        print(f"\n🔄 {model_name}")
        scores, rows, mean_acc, std_acc = train_one_model(
            model, model_name, X_all, y_enc, sub_all,
            class_names=le.classes_
        )
        summary_rows.append({
            "Model"   : model_name,
            "Mean_Acc": round(mean_acc, 4),
            "Std_Acc" : round(std_acc,  4),
        })
        detail_dict[model_name] = rows

    # ── Bảng tổng hợp ────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)\
                   .sort_values("Mean_Acc", ascending=False)\
                   .reset_index(drop=True)
    summary_df.insert(0, "Rank", range(1, len(summary_df) + 1))

    print("\n")
    print("=" * 55)
    print("  🏆 BẢNG XẾP HẠNG MODEL")
    print("=" * 55)
    print(summary_df.to_string(index=False))

    # ── Lưu CSV tổng hợp ─────────────────────────────────────────────────────
    summary_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ Bảng so sánh     → {summary_path}")

    # ── Lưu CSV chi tiết từng subject × model ────────────────────────────────
    subjects  = sorted(np.unique(sub_all))
    detail_df = pd.DataFrame({"Subject": [f"F{s}" for s in subjects]})
    for model_name, rows in detail_dict.items():
        df_tmp    = pd.DataFrame(rows)
        detail_df = detail_df.merge(df_tmp, on="Subject", how="left")

    detail_path = os.path.join(OUTPUT_DIR, "model_detail_per_subject.csv")
    detail_df.to_csv(detail_path, index=False)
    print(f"✅ Chi tiết subject → {detail_path}")

    # ── Classification report của model tốt nhất ─────────────────────────────
    best_name  = summary_df.iloc[0]["Model"]
    best_model = MODELS[best_name]
    logo       = LeaveOneGroupOut()
    scaler     = StandardScaler()
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in logo.split(X_all, y_enc, groups=sub_all):
        X_tr = scaler.fit_transform(X_all[train_idx])
        X_te = scaler.transform(X_all[test_idx])
        best_model.fit(X_tr, y_enc[train_idx])
        y_pred_all.extend(best_model.predict(X_te))
        y_true_all.extend(y_enc[test_idx])

    print(f"\n📋 Classification Report — {best_name}")
    print(classification_report(y_true_all, y_pred_all,
                                 target_names=le.classes_))

    return summary_df, detail_df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # Bước 1: Load cache
    X_all, y_all, sub_all = load_all_cache()

    if len(X_all) == 0:
        print("❌ Không có data — hãy chạy run_all_subjects.py trước!")
        exit()

    # Bước 2: Train tất cả model
    summary_df, detail_df = train_all_models(X_all, y_all, sub_all)
