# =============================================================================
# classify.py – Classification Pipeline
# Đọc features CSV từ eeg_pipeline.py → train & đánh giá 5 model
# Models: SVM, LDA, KNN, Random Forest, Naive Bayes (Gaussian)
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
)


# =============================================================================
# CONFIG
# =============================================================================

FEATURES_CSV = "/Users/audaodoantruc/Downloads/IU-MIBCI_4class/output/graph_features_all_epochs.csv"
OUTPUT_DIR   = "/Users/audaodoantruc/Downloads/IU-MIBCI_4class/output/"
N_FOLDS      = 5
RANDOM_STATE = 42


# =============================================================================
# MODELS
# =============================================================================

MODELS = {
    "SVM": SVC(
        kernel='rbf', C=1.0, gamma='scale',
        random_state=RANDOM_STATE
    ),
    "LDA": LinearDiscriminantAnalysis(
        solver='svd'
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=5, metric='euclidean'
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=None,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Naive Bayes": GaussianNB(),
}


# =============================================================================
# LOAD & PREPARE DATA
# =============================================================================

def load_features(csv_path):
    """Đọc features CSV, trả về X (array), y (array), feature_names (list)."""
    df = pd.read_csv(csv_path)
    y  = df['Label'].values
    X  = df.drop(columns=['Label']).values
    feature_names = df.drop(columns=['Label']).columns.tolist()
    print(f"Loaded: {csv_path}")
    print(f"  Samples  : {X.shape[0]}")
    print(f"  Features : {X.shape[1]}")
    print(f"  Classes  : {np.unique(y, return_counts=True)}")
    return X, y, feature_names


# =============================================================================
# TRAIN & EVALUATE
# =============================================================================

def evaluate_model(name, model, X_scaled, y_enc, le, cv, output_dir=""):
    """
    Chạy Stratified K-Fold CV cho 1 model.
    In kết quả, vẽ confusion matrix, trả về dict tóm tắt.
    """
    print(f"\n{'='*55}")
    print(f"  MODEL: {name}")
    print(f"{'='*55}")

    # Cross-validation scores
    cv_res = cross_validate(
        model, X_scaled, y_enc, cv=cv,
        scoring=['accuracy', 'f1_macro'],
        return_train_score=True
    )
    train_acc = cv_res['train_accuracy'].mean()
    train_std = cv_res['train_accuracy'].std()
    test_acc  = cv_res['test_accuracy'].mean()
    test_std  = cv_res['test_accuracy'].std()
    f1_macro  = cv_res['test_f1_macro'].mean()
    f1_std    = cv_res['test_f1_macro'].std()

    print(f"  Train Accuracy : {train_acc:.4f} ± {train_std:.4f}")
    print(f"  Test  Accuracy : {test_acc:.4f}  ± {test_std:.4f}")
    print(f"  Test  F1 Macro : {f1_macro:.4f}  ± {f1_std:.4f}")

    # Classification report
    y_pred = cross_val_predict(model, X_scaled, y_enc, cv=cv)
    print(f"\n  Classification Report:")
    print(classification_report(y_enc, y_pred, target_names=le.classes_,
                                 digits=4, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_enc, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)\
        .plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f"Confusion Matrix – {name}\n"
                 f"Test Acc = {test_acc:.4f} | F1 = {f1_macro:.4f}")
    plt.tight_layout()
    fig_name = f"{output_dir}confusion_matrix_{name.replace(' ', '_')}.png"
    plt.savefig(fig_name, dpi=150)
    plt.close()
    print(f"  Saved: {fig_name}")

    return {
        "Model"       : name,
        "Train_Acc"   : round(train_acc, 4),
        "Train_Std"   : round(train_std, 4),
        "Test_Acc"    : round(test_acc,  4),
        "Test_Std"    : round(test_std,  4),
        "F1_Macro"    : round(f1_macro,  4),
        "F1_Std"      : round(f1_std,    4),
    }


def plot_comparison(summary_df, output_dir=""):
    """Vẽ bar chart so sánh Test Accuracy và F1 của các model."""
    models = summary_df['Model'].tolist()
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, summary_df['Test_Acc'], width,
                   yerr=summary_df['Test_Std'], capsize=4,
                   label='Test Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, summary_df['F1_Macro'], width,
                   yerr=summary_df['F1_Std'], capsize=4,
                   label='F1 Macro', color='darkorange')

    ax.set_ylabel('Score')
    ax.set_title('Model Comparison – 3-Class EEG Motor Imagery\n'
                 f'(Stratified {N_FOLDS}-Fold CV)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.bar_label(bars1, fmt='%.3f', padding=3, fontsize=8)
    ax.bar_label(bars2, fmt='%.3f', padding=3, fontsize=8)

    plt.tight_layout()
    fig_name = f"{output_dir}model_comparison.png"
    plt.savefig(fig_name, dpi=150)
    plt.close()
    print(f"\nSaved comparison plot: {fig_name}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # --- 1. Load features ---
    X, y, feature_names = load_features(FEATURES_CSV)

    # --- 2. Encode & scale ---
    le       = LabelEncoder()
    y_enc    = le.fit_transform(y)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # --- 3. Train & evaluate mỗi model ---
    summary = []
    for name, model in MODELS.items():
        result = evaluate_model(name, model, X_scaled, y_enc, le, cv,
                                output_dir=OUTPUT_DIR)
        summary.append(result)

    # --- 4. Bảng tổng kết ---
    summary_df = pd.DataFrame(summary).sort_values('Test_Acc', ascending=False)
    summary_df = summary_df.reset_index(drop=True)

    print("\n" + "="*55)
    print("  TỔNG KẾT – XẾP HẠNG THEO TEST ACCURACY")
    print("="*55)
    print(summary_df.to_string(index=False))

    csv_out = f"{OUTPUT_DIR}model_summary.csv"
    summary_df.to_csv(csv_out, index=False)
    print(f"\n✅ Saved summary: {csv_out}")

    # --- 5. So sánh trực quan ---
    plot_comparison(summary_df, output_dir=OUTPUT_DIR)

    print("\n✅ Hoàn tất! Kiểm tra thư mục output để xem kết quả.")
