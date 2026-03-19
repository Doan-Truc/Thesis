# =============================================================================
# EEG Motor Imagery – Full Pipeline
# Preprocessing → Connectivity (MVAR + dTF) → Graph Theory → SVM Classification
# =============================================================================

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import solve
from sklearn.preprocessing import LabelEncoder


# =============================================================================
# CONFIG
# =============================================================================

EDF_PATH    = "/Users/audaodoantruc/Downloads/IU-MIBCI_4class/G1/F11_A.edf"
MARKER_PATH = "/Users/audaodoantruc/Downloads/IU-MIBCI_4class/G1/F11_A_intervalMarker.csv"
OUTPUT_DIR  = "/Users/audaodoantruc/Downloads/IU-MIBCI_4class/output/"

EEG_CHANNELS = [
    'Cz','Fz','Fp1','F7','F3','FC1','C3','FC5','FT9','T7',
    'CP5','CP1','P3','P7','PO9','O1','Pz','Oz','O2','PO10',
    'P8','P4','CP2','CP6','T8','FT10','FC6','C4','FC2',
    'F4','F8','Fp2'
]

# 4 class gốc trong file EDF/marker
CLASS_LABELS_RAW = ['Tay phải', 'Tay trái', 'Chân phải', 'Chân trái']
EVENT_ID         = {'Tay phải': 0, 'Tay trái': 1, 'Chân phải': 2, 'Chân trái': 3}

# 3 class sau khi gộp Chân phải + Chân trái → "Chân"
CLASS_LABELS     = ['Tay phải', 'Tay trái', 'Chân']
MERGE_LEGS       = {'Chân': ['Chân phải', 'Chân trái']}   # các class cần gộp

EPOCH_TMIN   = -0.5
EPOCH_TMAX   =  3.0
MVAR_ORDER   = 5
ALPHA_BAND   = (8, 13)
TOP_PERCENT  = 0.20


# =============================================================================
# SECTION 1 – PREPROCESSING
# =============================================================================

def load_and_preprocess(edf_path, eeg_channels):
    """Đọc EDF, lọc channel, bandpass, notch, ICA."""

    raw = mne.io.read_raw_edf(edf_path, preload=True)
    print("Sampling rate :", raw.info['sfreq'])
    print("Total channels:", len(raw.ch_names))

    # Chọn EEG channels
    raw_eeg = raw.copy().pick_channels(eeg_channels)
    print("EEG channels  :", raw_eeg.ch_names)

    # Bandpass filter 8–30 Hz
    raw_filt = raw_eeg.copy().filter(l_freq=8, h_freq=30)

    # Notch filter 50 Hz (powerline)
    raw_filt.notch_filter(freqs=50)

    # ICA artifact removal
    ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter="auto")
    ica.fit(raw_filt)
    # Ghi chú: không có EOG channel → bỏ qua auto-detect EOG artifact
    # Nếu có EOG: eog_indices, _ = ica.find_bads_eog(raw_filt); ica.exclude = eog_indices

    raw_clean = raw_filt.copy()
    ica.apply(raw_clean)

    print("Clean EEG shape:", raw_clean.get_data().shape)
    return raw_clean


def plot_eeg(raw_clean):
    """Vẽ tất cả EEG channels theo thời gian."""
    data = raw_clean.get_data()
    sfreq = raw_clean.info['sfreq']
    time = np.arange(data.shape[1]) / sfreq

    plt.figure(figsize=(14, 10))
    for i, ch in enumerate(raw_clean.ch_names):
        plt.subplot(len(raw_clean.ch_names), 1, i + 1)
        plt.plot(time, data[i])
        plt.ylabel(ch, rotation=0, labelpad=25)
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


# =============================================================================
# SECTION 2 – EPOCHING
# =============================================================================

def load_markers(marker_path):
    """Đọc file CSV marker."""
    marker_df = pd.read_csv(marker_path)
    print("Marker file loaded.")
    print("Columns:", marker_df.columns.tolist())
    print("Unique types:", marker_df['type'].unique())
    return marker_df


def create_epochs(raw_clean, marker_df, event_id, tmin, tmax):
    """Tạo epochs từ marker MI."""
    sfreq = raw_clean.info['sfreq']
    mi_markers = marker_df[marker_df['type'] == 'MI'].copy()

    events = []
    for _, row in mi_markers.iterrows():
        onset_sample = int(row['latency'] * sfreq)
        events.append([onset_sample, 0, row['marker_value']])
    events = np.array(events)

    epochs = mne.Epochs(
        raw_clean, events, event_id=event_id,
        tmin=tmin, tmax=tmax,
        preload=True, baseline=(None, 0), verbose=False
    )
    print("Number of epochs:", len(epochs))
    return epochs


def merge_epochs_labels(epochs, merge_map):
    """
    Gộp nhiều class thành 1 label mới trong metadata.
    merge_map: dict, vd {'Chân': ['Chân phải', 'Chân trái']}
    Trả về epochs gốc với metadata['merged_label'] đã được gán.
    """
    import pandas as pd
    # Xây reverse map: 'Chân phải' → 'Chân', 'Chân trái' → 'Chân'
    reverse = {}
    for new_label, old_labels in merge_map.items():
        for ol in old_labels:
            reverse[ol] = new_label

    # Gán merged_label dựa vào event_id → tên condition
    id_to_name = {v: k for k, v in epochs.event_id.items()}
    merged = [reverse.get(id_to_name.get(ev, ''), id_to_name.get(ev, ''))
              for ev in epochs.events[:, 2]]
    epochs.metadata = pd.DataFrame({'merged_label': merged},
                                   index=range(len(epochs)))
    return epochs


def plot_evoked(epochs, event_id):
    """Vẽ evoked response cho từng condition."""
    for condition in event_id.keys():
        subset = epochs[condition]
        if len(subset) > 0:
            subset.average().plot(picks='eeg', titles=f'Evoked – {condition}')
        else:
            print(f"No epochs for: {condition}")


# =============================================================================
# SECTION 3 – MVAR + dTF CONNECTIVITY
# =============================================================================

def estimate_mvar_yw(X, order):
    """
    MVAR Yule–Walker cho toàn bộ class (multi-epoch ghép lại).
    X: (n_channels, n_samples)
    """
    n_channels, n_samples = X.shape
    R = np.zeros((order + 1, n_channels, n_channels))

    for k in range(order + 1):
        denom = n_samples - k
        if denom <= 0:
            raise ValueError("Not enough samples for MVAR")
        R[k] = (X[:, k:] @ X[:, :n_samples - k].T) / denom

    R0 = np.hstack([R[i] for i in range(1, order + 1)])
    T  = np.vstack([np.hstack([R[abs(i - j)] for j in range(1, order + 1)])
                    for i in range(1, order + 1)])

    A_vec      = solve(T, R0.T).T
    A_matrices = A_vec.reshape(order, n_channels, n_channels)
    E = R[0] - sum(A_matrices[p] @ R[p + 1] for p in range(order))
    return A_matrices, E


def estimate_mvar_yw_single(X, order):
    """MVAR Yule–Walker cho 1 epoch đơn lẻ. X: (n_channels, n_times)"""
    n_channels, n_samples = X.shape
    X = X - X.mean(axis=1, keepdims=True)   # demean
    R = np.zeros((order + 1, n_channels, n_channels))

    for k in range(order + 1):
        denom = n_samples - k
        if denom <= 0:
            raise ValueError("Epoch quá ngắn")
        R[k] = (X[:, k:] @ X[:, :n_samples - k].T) / denom

    R0 = np.hstack([R[i] for i in range(1, order + 1)])
    T  = np.vstack([np.hstack([R[abs(i - j)] for j in range(1, order + 1)])
                    for i in range(1, order + 1)])

    A_vec      = solve(T, R0.T).T
    A_matrices = A_vec.reshape(order, n_channels, n_channels)
    return A_matrices


def compute_dtf(A_matrices, fs=128, n_freqs=200):
    """Tính Directed Transfer Function (dTF) từ MVAR coefficients."""
    P, n, _ = A_matrices.shape
    freqs   = np.linspace(0, fs / 2, n_freqs)
    dtf     = np.zeros((n, n, n_freqs), float)
    I       = np.eye(n, dtype=complex)

    for fi, f in enumerate(freqs):
        Af = I.copy()
        for p in range(P):
            Af -= A_matrices[p] * np.exp(-2j * np.pi * f * (p + 1) / fs)
        try:
            Hf = np.linalg.inv(Af)
        except np.linalg.LinAlgError:
            Hf = np.linalg.inv(Af + 1e-8 * np.eye(n))

        H2 = np.abs(Hf) ** 2
        denom = np.sum(H2, axis=0, keepdims=True)
        denom[denom == 0] = 1e-16
        dtf[:, :, fi] = H2 / denom

    return freqs, dtf


def apply_threshold(weighted_matrix, top_percent=0.05):
    """Giữ top X% connections mạnh nhất → binary matrix."""
    mat  = weighted_matrix.copy()
    flat = mat[~np.eye(mat.shape[0], dtype=bool)]
    threshold_val = np.percentile(flat, (1 - top_percent) * 100)
    binary = (mat >= threshold_val).astype(int)
    np.fill_diagonal(binary, 0)
    return binary, threshold_val


def process_class_connectivity(epochs_class, class_name, channel_names, fs,
                                order=5, alpha_band=(8, 13), output_dir=""):
    """
    Chạy MVAR → dTF → weighted connectivity matrix cho 1 class.
    Lưu CSV + vẽ heatmap.
    """
    print(f"\n===== PROCESSING CLASS: {class_name} =====")
    data = epochs_class.get_data()           # (n_epochs, n_channels, n_times)
    eeg  = data.reshape(-1, data.shape[1]).T # ghép epochs → (n_channels, n_samples)
    print("Merged EEG shape:", eeg.shape)

    A, _ = estimate_mvar_yw(eeg, order=order)
    freqs, dtf = compute_dtf(A, fs=fs)

    alpha_idx = np.where((freqs >= alpha_band[0]) & (freqs <= alpha_band[1]))[0]
    weighted_matrix = np.mean(dtf[:, :, alpha_idx], axis=2)
    np.fill_diagonal(weighted_matrix, 0)

    # Lưu CSV
    csv_name = f"{output_dir}dTF_alpha_{class_name.replace(' ', '_')}.csv"
    pd.DataFrame(weighted_matrix, index=channel_names, columns=channel_names)\
      .to_csv(csv_name, float_format="%.6f")
    print("Saved:", csv_name)

    # Vẽ heatmap
    n_ch = len(channel_names)
    plt.figure(figsize=(9, 7))
    plt.imshow(weighted_matrix, cmap='viridis', origin='lower')
    plt.colorbar(label='dTF (alpha)')
    plt.xticks(np.arange(n_ch), channel_names, rotation=90)
    plt.yticks(np.arange(n_ch), channel_names)
    plt.title(f"Connectivity dTF Alpha (8–13 Hz) – {class_name}")
    plt.tight_layout()
    plt.show()

    return weighted_matrix


# =============================================================================
# SECTION 4 – GRAPH THEORY METRICS
# =============================================================================

def compute_graph_metrics(binary_matrix, channel_names):
    """Tính các graph metrics từ binary adjacency matrix."""
    G = nx.from_numpy_array(binary_matrix, create_using=nx.DiGraph)
    G = nx.relabel_nodes(G, {i: ch for i, ch in enumerate(channel_names)})
    G_und = G.to_undirected()

    # Node Strength: tổng số cạnh kết nối (với binary matrix = Degree,
    # giữ riêng để dễ thay bằng weighted matrix sau này)
    node_strength = {ch: float(np.sum(binary_matrix[i]))
                     for i, ch in enumerate(channel_names)}

    # Local Efficiency: tính trên subgraph của từng node
    local_eff = {n: nx.local_efficiency(G_und.subgraph(
                     list(G_und.neighbors(n)) + [n]))
                 for n in G_und.nodes()}

    metrics = {
        "Degree"           : dict(G.degree()),
        "In-Degree"        : dict(G.in_degree()),
        "Out-Degree"       : dict(G.out_degree()),
        "Clustering"       : nx.clustering(G_und),
        "Global_Efficiency": nx.global_efficiency(G_und),
        "Local_Efficiency" : local_eff,
        "Node_Strength"    : node_strength,
    }
    return G, metrics


def plot_metrics(metrics, channel_names, class_name, output_dir=""):
    """Vẽ bar chart: Degree, Clustering, Local Efficiency, Node Strength.
       Global Efficiency là scalar nên hiển thị trên title."""
    bar_items = [
        ("Degree",           "Degree",                "steelblue"),
        ("Clustering",       "Clustering Coefficient","darkorange"),
        ("Local_Efficiency", "Local Efficiency",      "seagreen"),
        ("Node_Strength",    "Node Strength",         "mediumpurple"),
    ]

    fig, axes = plt.subplots(len(bar_items), 1, figsize=(13, 14))
    global_eff = metrics["Global_Efficiency"]
    fig.suptitle(
        f"Graph Metrics – {class_name} (Top 20% threshold)\n"
        f"Global Efficiency = {global_eff:.4f}",
        fontsize=13
    )

    for ax, (key, ylabel, color) in zip(axes, bar_items):
        values = [metrics[key][ch] for ch in channel_names]
        ax.bar(channel_names, values, color=color)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(channel_names)))
        ax.set_xticklabels(channel_names, rotation=90)

    plt.tight_layout()
    fig_name = f"{output_dir}graph_metrics_{class_name.replace(' ', '_')}.png"
    plt.savefig(fig_name, dpi=150)
    plt.show()
    print(f"  Saved figure: {fig_name}")


def plot_network(G, metrics, class_name, output_dir=""):
    """Vẽ circular brain network graph."""
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.circular_layout(G)

    node_size  = [metrics["Degree"][n] * 80 + 100 for n in G.nodes()]
    node_color = [metrics["Node_Strength"][n] for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size,
                           node_color=node_color, cmap=plt.cm.YlOrRd, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray',
                           arrows=True, arrowsize=10, alpha=0.5, width=0.8)

    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.YlOrRd,
        norm=plt.Normalize(vmin=min(node_color), vmax=max(node_color))
    )
    fig.colorbar(sm, ax=ax, label="Node Strength")
    ax.set_title(f"Brain Network – {class_name} (Top 20%)")
    ax.axis('off')
    plt.tight_layout()

    fig_name = f"{output_dir}network_{class_name.replace(' ', '_')}.png"
    plt.savefig(fig_name, dpi=150)
    plt.show()
    print(f"  Saved figure: {fig_name}")


def save_metrics_csv(metrics, channel_names, class_name, output_dir=""):
    """Lưu graph metrics ra CSV."""
    df = pd.DataFrame({
        "Channel"         : channel_names,
        "Degree"          : [metrics["Degree"][ch]          for ch in channel_names],
        "In_Degree"       : [metrics["In-Degree"][ch]       for ch in channel_names],
        "Out_Degree"      : [metrics["Out-Degree"][ch]      for ch in channel_names],
        "Node_Strength"   : [metrics["Node_Strength"][ch]   for ch in channel_names],
        "Clustering"      : [metrics["Clustering"][ch]      for ch in channel_names],
        "Local_Efficiency": [metrics["Local_Efficiency"][ch] for ch in channel_names],
    })
    csv_name = f"{output_dir}graph_metrics_{class_name.replace(' ', '_')}.csv"
    df.to_csv(csv_name, index=False)
    print(f"  Saved CSV         : {csv_name}")
    print(f"  Global Efficiency : {metrics['Global_Efficiency']:.4f}")
    return df


def run_graph_pipeline(results, class_labels, channel_names, output_dir=""):
    """Chạy toàn bộ graph pipeline cho 4 class."""
    graph_results = {}

    for label in class_labels:
        print(f"\n========== GRAPH: {label} ==========")
        weighted_matrix = results[label]

        binary_matrix, thr_val = apply_threshold(weighted_matrix, top_percent=TOP_PERCENT)
        print(f"  Threshold value: {thr_val:.6f} | Edges kept: {binary_matrix.sum()}")

        # Lưu binary matrix
        bin_csv = f"{output_dir}dTF_alpha_{label.replace(' ', '_')}_binary_top20.csv"
        pd.DataFrame(binary_matrix, index=channel_names, columns=channel_names)\
          .to_csv(bin_csv)
        print(f"  Saved binary: {bin_csv}")

        G, metrics = compute_graph_metrics(binary_matrix, channel_names)
        plot_metrics(metrics, channel_names, label, output_dir)
        plot_network(G, metrics, label, output_dir)
        df_metrics = save_metrics_csv(metrics, channel_names, label, output_dir)

        graph_results[label] = {
            "G": G, "metrics": metrics,
            "binary": binary_matrix, "df": df_metrics,
        }

    print("\n✅ Hoàn tất pipeline: MVAR → dTF → Graph Theory cho 4 lớp.")
    return graph_results


# =============================================================================
# SECTION 5 – FEATURE EXTRACTION
# =============================================================================

def extract_features_single_epoch(X, order=5, fs=128, top_percent=0.20):
    """
    Trích xuất feature vector từ 1 epoch.
    X: (n_channels, n_times)
    Returns: 1D feature vector, hoặc None nếu lỗi.
    """
    try:
        A = estimate_mvar_yw_single(X, order)
    except Exception:
        return None

    freqs, dtf = compute_dtf(A, fs=fs)

    alpha_idx = np.where((freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1]))[0]
    W = np.mean(dtf[:, :, alpha_idx], axis=2)
    np.fill_diagonal(W, 0)

    binary, _ = apply_threshold(W, top_percent=top_percent)

    G     = nx.from_numpy_array(binary, create_using=nx.DiGraph)
    G_und = G.to_undirected()

    degree       = np.array([d for _, d in G.degree()])
    in_degree    = np.array([d for _, d in G.in_degree()])
    out_degree   = np.array([d for _, d in G.out_degree()])
    node_strength= np.sum(binary, axis=1).astype(float)
    clustering   = np.array(list(nx.clustering(G_und).values()))
    global_eff   = np.array([nx.global_efficiency(G_und)])
    local_eff    = np.array([
        nx.local_efficiency(G_und.subgraph(list(G_und.neighbors(n)) + [n]))
        for n in G_und.nodes()
    ])

    feature_vec = np.concatenate([
        degree,
        in_degree,
        out_degree,
        node_strength,
        clustering,
        local_eff,
        global_eff,
    ])
    return feature_vec


def build_feature_matrix(epochs, class_labels, order=5, fs=128, top_percent=0.20):
    """Build dataset X, y từ tất cả epochs (hỗ trợ merged_label)."""
    X_list, y_list = [], []

    # Nếu đã gộp label thì dùng metadata['merged_label']
    use_metadata = (epochs.metadata is not None and 'merged_label' in epochs.metadata.columns)

    for label in class_labels:
        if use_metadata:
            mask = epochs.metadata['merged_label'] == label
            data = epochs.get_data()[mask.values]
        else:
            data = epochs[label].get_data()
        print(f"\nProcessing [{label}]: {data.shape[0]} epochs")

        for i in range(data.shape[0]):
            feat = extract_features_single_epoch(data[i], order=order,
                                                 fs=fs, top_percent=top_percent)
            if feat is not None:
                X_list.append(feat)
                y_list.append(label)
            else:
                print(f"  ⚠️  Epoch {i} bị lỗi, bỏ qua")

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"\nFeature matrix X: {X.shape}")
    print(f"Labels y        : {np.unique(y, return_counts=True)}")
    return X, y



# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # --- 1. Preprocessing ---
    raw_clean = load_and_preprocess(EDF_PATH, EEG_CHANNELS)
    # plot_eeg(raw_clean)   # bỏ comment nếu muốn xem EEG thô

    # --- 2. Epoching ---
    marker_df = load_markers(MARKER_PATH)
    epochs    = create_epochs(raw_clean, marker_df, EVENT_ID, EPOCH_TMIN, EPOCH_TMAX)
    # plot_evoked(epochs, EVENT_ID)   # bỏ comment nếu muốn xem evoked

    channel_names = epochs.info['ch_names']
    fs            = int(epochs.info['sfreq'])

    # --- 3. Gộp class Chân phải + Chân trái → "Chân" ---
    epochs = merge_epochs_labels(epochs, MERGE_LEGS)

    # --- 4. MVAR + dTF Connectivity (per class) ---
    # Với class gộp ("Chân"), lấy epochs qua metadata
    results = {}
    for label in CLASS_LABELS:
        if label in MERGE_LEGS:
            mask = epochs.metadata['merged_label'] == label
            ep_subset = epochs[mask.values]
        else:
            ep_subset = epochs[label]
        results[label] = process_class_connectivity(
            ep_subset, label, channel_names, fs,
            order=MVAR_ORDER, alpha_band=ALPHA_BAND, output_dir=OUTPUT_DIR
        )

    # --- 5. Graph Theory ---
    graph_results = run_graph_pipeline(results, CLASS_LABELS, channel_names, OUTPUT_DIR)

    # --- 6. Feature Extraction → lưu CSV (chạy classify.py để train model) ---
    X, y = build_feature_matrix(epochs, CLASS_LABELS, order=MVAR_ORDER,
                                 fs=fs, top_percent=TOP_PERCENT)

    ch = epochs.info['ch_names']
    feature_names = (
        [f"degree_{c}"        for c in ch] +
        [f"in_degree_{c}"     for c in ch] +
        [f"out_degree_{c}"    for c in ch] +
        [f"node_strength_{c}" for c in ch] +
        [f"clustering_{c}"    for c in ch] +
        [f"local_eff_{c}"     for c in ch] +
        ["global_efficiency"]
    )
    feat_df = pd.DataFrame(X, columns=feature_names)
    feat_df.insert(0, 'Label', y)
    feat_path = f"{OUTPUT_DIR}graph_features_all_epochs.csv"
    feat_df.to_csv(feat_path, index=False)
    print(f"\n✅ Features saved → {feat_path}")
    print("   Chạy classify.py để train và đánh giá các model.")
