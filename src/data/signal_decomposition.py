"""
CEEMDAN signal decomposition for freight-rate forecasting.

Researcher 1 - Day 2

Pipeline:
    Freight rate series
        ↓
    CEEMDAN decomposition
        ↓
    Intrinsic Mode Functions (IMFs)
        ↓
    Gaussian filtering of high-frequency IMFs
        ↓
    Feature extraction:
        - trend_slope
        - volatility_amplitude
        - dominant_cycle_period
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import periodogram
from PyEMD import CEEMDAN


def decompose_signal(
    signal: np.ndarray,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Decompose a 1-D signal into Intrinsic Mode Functions using CEEMDAN.

    Args:
        signal: 1-D freight-rate time series.
        random_seed: Seed for reproducibility.

    Returns:
        Array with shape (n_imfs, n_samples).
    """
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if len(signal) < 20:
        raise ValueError("Signal must contain at least 20 observations.")

    if not np.isfinite(signal).all():
        raise ValueError("Signal contains NaN or infinite values.")

    ceemdan = CEEMDAN()
    ceemdan.noise_seed(random_seed)

    imfs = ceemdan.ceemdan(signal)

    return np.asarray(imfs)


def filter_high_frequency_imfs(
    imfs: np.ndarray,
    sigma: float = 2.0,
    high_frequency_count: int = 2,
) -> np.ndarray:
    """
    Apply Gaussian smoothing to the first/high-frequency IMFs.

    The highest-frequency components are normally the first IMFs
    produced by CEEMDAN.

    Args:
        imfs: IMF matrix, shape (n_imfs, n_samples).
        sigma: Gaussian filter strength.
        high_frequency_count: Number of leading IMFs to smooth.

    Returns:
        Filtered IMF matrix.
    """
    filtered = np.asarray(imfs, dtype=float).copy()

    count = min(high_frequency_count, len(filtered))

    for i in range(count):
        filtered[i] = gaussian_filter1d(
            filtered[i],
            sigma=sigma,
        )

    return filtered


def extract_features(
    signal: np.ndarray,
    imfs: np.ndarray,
) -> dict:
    """
    Extract CEEMDAN-derived forecasting features.

    Features:
        trend_slope:
            Linear slope of the reconstructed low-frequency/trend signal.

        volatility_amplitude:
            Standard deviation of the high-frequency component.

        dominant_cycle_period:
            Dominant period detected from the reconstructed signal.
    """
    signal = np.asarray(signal, dtype=float)
    imfs = np.asarray(imfs, dtype=float)

    if imfs.ndim != 2:
        raise ValueError("IMFs must be a 2-D array.")

    n_imfs = imfs.shape[0]

    # CEEMDAN's final component represents the residual/trend.
    trend = imfs[-1]

    # Linear trend slope.
    x = np.arange(len(trend))
    trend_slope = float(np.polyfit(x, trend, 1)[0])

    # High-frequency component = first few IMFs.
    high_count = min(2, n_imfs)

    if high_count > 0:
        high_frequency = np.sum(
            imfs[:high_count],
            axis=0,
        )
        volatility_amplitude = float(
            np.std(high_frequency)
        )
    else:
        volatility_amplitude = 0.0

    # Detect dominant cycle from the non-trend signal.
    detrended = signal - trend

    if np.allclose(detrended, 0):
        dominant_cycle_period = 0.0
    else:
        frequencies, power = periodogram(detrended)

        # Ignore zero frequency.
        valid = frequencies > 0

        if not np.any(valid):
            dominant_cycle_period = 0.0
        else:
            valid_frequencies = frequencies[valid]
            valid_power = power[valid]

            dominant_frequency = valid_frequencies[
                np.argmax(valid_power)
            ]

            dominant_cycle_period = float(
                1.0 / dominant_frequency
            )

    return {
        "trend_slope": trend_slope,
        "volatility_amplitude": volatility_amplitude,
        "dominant_cycle_period": dominant_cycle_period,
    }


def analyze_freight_series(
    df: pd.DataFrame,
    target_column: str = "freight_rate",
    gaussian_sigma: float = 2.0,
    high_frequency_count: int = 2,
) -> dict:
    """
    Run the complete CEEMDAN analysis on a freight-rate DataFrame.

    Returns:
        Dictionary containing IMFs, filtered IMFs and extracted features.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Column '{target_column}' not found in dataframe."
        )

    signal = (
        pd.to_numeric(
            df[target_column],
            errors="coerce",
        )
        .dropna()
        .to_numpy()
    )

    imfs = decompose_signal(signal)

    filtered_imfs = filter_high_frequency_imfs(
        imfs,
        sigma=gaussian_sigma,
        high_frequency_count=high_frequency_count,
    )

    features = extract_features(
        signal,
        filtered_imfs,
    )

    return {
        "imfs": imfs,
        "filtered_imfs": filtered_imfs,
        "features": features,
    }


if __name__ == "__main__":
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    data_path = (
        root
        / "data"
        / "processed"
        / "merged_market_data.csv"
    )

    if not data_path.exists():
        print("Processed dataset not found.")
        raise SystemExit(1)

    df = pd.read_csv(data_path)

    result = analyze_freight_series(df)

    print("\nCEEMDAN DECOMPOSITION COMPLETE")
    print("=" * 45)

    print(f"Samples: {len(df)}")
    print(f"Number of IMFs: {len(result['imfs'])}")

    print("\nExtracted features:")
    for name, value in result["features"].items():
        print(f"  {name}: {value:.6f}")