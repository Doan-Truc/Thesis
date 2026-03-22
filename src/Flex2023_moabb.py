# =============================================================================
# Flex2023_moabb.py
# Motor Imagery dataset — International University (VNU-HCM)
# Hỗ trợ cấu trúc thư mục: G1 (F10, F11), G2, G3
# =============================================================================

import os
import numpy as np
import pandas as pd
import mne
from moabb.datasets.base import BaseDataset


# =============================================================================
# CONFIG
# =============================================================================

DATA_ROOT = "/Users/audaodoantruc/Downloads/IU-MIBCI_4class/data"

# Mỗi group có thư mục riêng
ROOTS = {
    "G1": os.path.join(DATA_ROOT, "G1"),
    "G2": os.path.join(DATA_ROOT, "G2"),
    "G3": os.path.join(DATA_ROOT, "G3"),
}

# Subject thuộc group nào
# G1: F10, F11 (CSV marker riêng), F12 (có _A + run) — F13 bỏ qua vì không có _A
# G2: F25, F26, F27, F29
# G3: F14, F15, F16, F18, F20, F22, F24
SUBJECT_GROUP = {
    10: "G1", 11: "G1", 12: "G1",
    25: "G2", 26: "G2", 27: "G2", 29: "G2",
    14: "G3", 15: "G3", 16: "G3", 18: "G3",
    20: "G3", 22: "G3", 24: "G3",
}

LIST_SUBJECTS   = list(SUBJECT_GROUP.keys())
ALL_EVENTS_IDS  = dict(right_hand=1, left_hand=2, right_foot=3, left_foot=4)
EEG_CH_NAMES    = [
    'Cz',  'Fz',  'Fp1', 'F7',  'F3',
    'FC1', 'C3',  'FC5', 'FT9', 'T7',
    'CP5', 'CP1', 'P3',  'P7',  'PO9',
    'O1',  'Pz',  'Oz',  'O2',  'PO10',
    'P8',  'P4',  'CP2', 'CP6', 'T8',
    'FT10','FC6', 'C4',  'FC2', 'F4',
    'F8',  'Fp2'
]
FS = 128


# =============================================================================
# CLASS
# =============================================================================

class Flex2023_moabb(BaseDataset):
    """Motor Imagery MOABB dataset — IU VNU-HCM"""

    def __init__(self):
        super().__init__(
            subjects        = LIST_SUBJECTS,
            sessions_per_subject = 1,
            events          = ALL_EVENTS_IDS,
            code            = "Flex2023",
            interval        = [4, 8],
            paradigm        = "imagery",
            doi             = "",
        )
        self.runs = -1  # -1 = lấy tất cả runs


    def _flow(self, raw0, stim):
        """Ghép EEG (32 ch) + Stim (1 ch) thành MNE Raw object."""
        data = raw0.get_data(picks=EEG_CH_NAMES)
        data = np.vstack([data, stim.reshape(1, -1)])

        ch_types = ["eeg"] * 32 + ["stim"]
        ch_names = EEG_CH_NAMES + ["Stim"]
        info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=FS)
        raw  = mne.io.RawArray(data=data, info=info, verbose=False)

        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage)
        return raw


    def _get_single_subject_data(self, subject):
        """Load và concat tất cả runs của 1 subject."""
        list_edf = self.data_path(subject)

        if len(list_edf) == 0:
            raise FileNotFoundError(f"Không tìm thấy file EDF cho subject F{subject}")

        # Chọn runs
        if subject in [10, 11] or self.runs == -1:
            list_edf_select = list_edf
        else:
            list_edf_select = [p for p in list_edf if f"run{self.runs}" in p]

        print(f"  F{subject}: {len(list_edf_select)} file EDF")
        for f in list_edf_select:
            print(f"    → {os.path.basename(f)}")

        # Concat runs
        list_raw = []
        for edf_path in list_edf_select:
            raw0    = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            stim    = get_stim_data(raw0, subject)
            raw_run = self._flow(raw0, stim)
            list_raw.append(raw_run)

        raw = mne.concatenate_raws(list_raw)
        return {"0": {"0": raw}}


    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):
        """
        Tìm tất cả file EDF của subject trong đúng thư mục group.
        Chỉ lấy file có '_A' trong tên (bỏ qua F13 vì không có _A).
        """
        group    = SUBJECT_GROUP.get(subject)
        if group is None:
            print(f"  ⚠️  Subject F{subject} không thuộc group nào — bỏ qua")
            return []

        root_dir = ROOTS[group]
        list_edf = []

        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if not file.endswith(".edf"):
                    continue
                if ".md" in file:
                    continue
                # Phải có F{subject}_ VÀ _A trong tên
                if f"F{subject}_" in file and "_A" in file:
                    list_edf.append(os.path.join(root, file))

        return sorted(list_edf)


# =============================================================================
# MARKER HANDLING
# =============================================================================

def get_stim_data(edf_raw, subject):
    """
    Lấy stim channel đúng theo từng subject:
    - F10, F11: EDF bị sai → tìm file CSV trong G1 để fix
    - F12+    : đọc thẳng kênh MarkerValueInt
    """
    if subject in [10, 11]:
        # Tìm file CSV tương ứng: F10_A_intervalMarker.csv hoặc F11_A_intervalMarker.csv
        path_csv = None
        for root, dirs, files in os.walk(ROOTS["G1"]):
            for file in files:
                if (file.endswith(".csv") and
                    "intervalMarker" in file and
                    f"F{subject}_A" in file):
                        path_csv = os.path.join(root, file)

        if path_csv is None:
            print(f"  ⚠️  F{subject}: không tìm thấy CSV → fix không có CSV")
        else:
            print(f"  F{subject}: dùng CSV → {os.path.basename(path_csv)}")

        return fix_stim(edf_raw, path_csv=path_csv)

    else:
        return edf_raw.get_data(picks=["MarkerValueInt"], units='uV')[0]


def fix_stim(edf_raw, path_csv=None):
    """
    Fix stim cho F10/F11 — procedure cũ ghi sai MarkerValueInt.

    320 markers chia 3 nhóm:
      [0  →119]: 120 biocalib → offset +20
      [120→139]: 20  kines    → offset +10
      [140→319]: 180 MI       → nhãn 1/2/3/4  ← phần dùng để train
    """
    markerIndex    = edf_raw.get_data(picks=["MarkerIndex"],    units='uV')[0]
    markerValueInt = edf_raw.get_data(picks=["MarkerValueInt"], units='uV')[0]

    markers = np.where(markerIndex != 0)[0]  # vị trí (sample) có event
    stim    = np.zeros_like(markerValueInt)

    if path_csv is not None:
        df = pd.read_csv(path_csv)

    for i, sample_pos in enumerate(markers):
        base = df.loc[i, "marker_value"] if path_csv is not None else markerValueInt[sample_pos]

        if 0 <= i < 120:        # biocalib
            stim[sample_pos] = base + 20
        elif 120 <= i < 140:    # kines
            stim[sample_pos] = base + 10
        else:                   # MI
            stim[sample_pos] = 1 if base == 0 else base + 1

    return stim


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("Subjects:", LIST_SUBJECTS)
    print("Groups  :", SUBJECT_GROUP)
    print()

    dataset = Flex2023_moabb()

    # Test load 1 subject
    test_sub = 15
    print(f"Test load subject F{test_sub}...")
    files = dataset.data_path(test_sub)
    print(f"  Files tìm được: {[os.path.basename(f) for f in files]}")

    data = dataset._get_single_subject_data(test_sub)
    raw  = data["0"]["0"]
    print(f"  Raw shape : {raw.get_data().shape}")
    print(f"  Channels  : {raw.ch_names}")
    print(f"  Duration  : {raw.times[-1]:.1f}s")
    print("✅ Load OK")
