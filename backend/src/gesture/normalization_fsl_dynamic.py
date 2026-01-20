# src/gesture/normalization_fsl_dynamic.py

import numpy as np
import json


class DynamicLandmarkNormalizer:
    """
    Normalizer for dynamic FSL sequences
    Shape expected: (num_sequences, seq_len, num_features)
    Example feature size: 126 (2 hands × 21 landmarks × 3 coords)
    """

    def __init__(self, method="zscore", wrist_relative=False):
        self.method = method
        self.wrist_relative = wrist_relative
        self.mean = None
        self.std = None
        self.min = None
        self.max = None

    # -------------------------------------------------
    # 🧠 FIT (TRAINING DATA ONLY)
    # -------------------------------------------------
    def fit(self, sequences):
        """
        Fit normalization parameters using training sequences
        sequences: np.ndarray (N, T, F)
        """

        assert sequences.ndim == 3, "Expected shape (N, T, F)"

        # Flatten time & samples → (N*T, F)
        flat = sequences.reshape(-1, sequences.shape[-1])

        if self.method == "zscore":
            self.mean = np.mean(flat, axis=0)
            self.std = np.std(flat, axis=0)
            self.std = np.where(self.std == 0, 1, self.std)

        elif self.method == "minmax":
            self.min = np.min(flat, axis=0)
            self.max = np.max(flat, axis=0)
            range_val = self.max - self.min
            self.max = np.where(range_val == 0, self.min + 1, self.max)

    # -------------------------------------------------
    # 🔄 TRANSFORM (FRAME-BY-FRAME)
    # -------------------------------------------------
    def transform(self, sequence):
        """
        Normalize a single sequence
        sequence: np.ndarray (T, F)
        """

        sequence = sequence.copy()

        # Wrist-relative normalization (per frame)
        if self.wrist_relative:
            for t in range(sequence.shape[0]):
                wrist_x = sequence[t, 0]
                wrist_y = sequence[t, 1]
                wrist_z = sequence[t, 2]

                for i in range(0, sequence.shape[1], 3):
                    sequence[t, i]     -= wrist_x
                    sequence[t, i + 1] -= wrist_y
                    sequence[t, i + 2] -= wrist_z

        # Statistical normalization
        if self.method == "zscore" and self.mean is not None:
            sequence = (sequence - self.mean) / self.std

        elif self.method == "minmax" and self.min is not None:
            sequence = (sequence - self.min) / (self.max - self.min)

        return sequence

    # -------------------------------------------------
    # 💾 SAVE / LOAD
    # -------------------------------------------------
    def save(self, filepath):
        params = {
            "method": self.method,
            "wrist_relative": self.wrist_relative,
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
            "min": self.min.tolist() if self.min is not None else None,
            "max": self.max.tolist() if self.max is not None else None,
        }

        with open(filepath, "w") as f:
            json.dump(params, f)

        print(f"✅ Dynamic normalizer saved to {filepath}")

    def load(self, filepath):
        with open(filepath, "r") as f:
            params = json.load(f)

        self.method = params["method"]
        self.wrist_relative = params["wrist_relative"]
        self.mean = np.array(params["mean"]) if params["mean"] else None
        self.std = np.array(params["std"]) if params["std"] else None
        self.min = np.array(params["min"]) if params["min"] else None
        self.max = np.array(params["max"]) if params["max"] else None

        print(
            f"✅ Loaded dynamic normalizer "
            f"(method={self.method}, wrist_relative={self.wrist_relative})"
        )


# -------------------------------------------------
# 🏭 FACTORY
# -------------------------------------------------
def create_dynamic_normalizer(method="zscore", wrist_relative=False):
    return DynamicLandmarkNormalizer(
        method=method,
        wrist_relative=wrist_relative
    )
