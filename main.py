"""
=============================================================================
EDS Final Project — VIB-01: Natural Frequency Resonance
Course: Computer Programming | Academic Year: 2026
Pillar 10: Vibration & Noise Control | Topic 01
=============================================================================
Pipeline: Data Ingestion → Cleaning → Analysis → Visualization
Libraries: NumPy, Pandas, SciPy, Matplotlib, Plotly
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis, pearsonr
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "input_path":        "data/no_fault.csv",
    "output_clean_path": "data/dataset_cleaned.csv",
    "output_dir":        "outputs/",
    "sample_rate_hz":    10_000,
    "unique_filter_col": "gear_fault_desc",
    "unique_filter_val": "No fault",
    "amplitude_col":     "sensor1",
    "time_col":          "time_x",
    "group_col":         "gear_fault_desc",
}


# =============================================================================
# CLASS: VibrationPipeline
# =============================================================================

class VibrationPipeline:
    """
    Production-grade OOP pipeline for VIB-01 Natural Frequency Resonance.

    Modules
    -------
    1. ingest()          — Load and validate the raw CSV dataset
    2. clean()           — Detect/remove nulls, duplicates, type errors, apply filter
    3. analyze()         — NumPy statistics, FFT, correlation, comparative analysis
    4. visualize_static()  — 3 static matplotlib charts saved to outputs/
    5. visualize_animated() — 2 animated charts (Matplotlib + Plotly)
    6. run()             — Orchestrate the full pipeline
    """

    def __init__(self, config: dict):
        self.config      = config
        self.raw_df      = None
        self.clean_df    = None
        self.stats       = {}
        self.fft_result  = {}
        os.makedirs(self.config["output_dir"], exist_ok=True)
        os.makedirs("data", exist_ok=True)

    # =========================================================================
    # MODULE 1 — DATA INGESTION
    # =========================================================================

    def ingest(self) -> None:
        """
        Load CSV from disk with full error handling.
        Validates the file exists, is non-empty, and contains expected columns.
        """
        print("\n" + "="*60)
        print("MODULE 1 — DATA INGESTION")
        print("="*60)

        filepath = self.config["input_path"]

        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(
                    f"Dataset not found at '{filepath}'.\n"
                    "Please download from Kaggle and place it in data/"
                )

            self.raw_df = pd.read_csv(filepath)

            if self.raw_df.empty:
                raise ValueError("Loaded dataset is empty (0 rows).")

            print(f"[OK] Loaded dataset: {filepath}")
            print(f"     Shape      : {self.raw_df.shape[0]} rows × {self.raw_df.shape[1]} cols")
            print(f"     Columns    : {list(self.raw_df.columns)}")
            print(f"     Memory     : {self.raw_df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            print(f"     Dtypes     :\n{self.raw_df.dtypes.to_string()}")

        except FileNotFoundError as e:
            print(f"[ERROR] File not found — {e}")
            self._generate_synthetic_data()

        except pd.errors.ParserError as e:
            print(f"[ERROR] CSV parse error — {e}")
            sys.exit(1)

        except ValueError as e:
            print(f"[ERROR] Data error — {e}")
            sys.exit(1)

        except Exception as e:
            print(f"[ERROR] Unexpected error during ingestion — {e}")
            sys.exit(1)

    def _generate_synthetic_data(self) -> None:
        """
        Fallback: generate realistic synthetic vibration data so the pipeline
        can be demonstrated without the Kaggle dataset.
        Replace this with your real dataset download.
        """
        print("\n[INFO] Generating synthetic vibration dataset for demonstration...")

        np.random.seed(42)
        n        = 50_000
        fs       = self.config["sample_rate_hz"]
        t        = np.linspace(0, n / fs, n)

        # Simulate two conditions: Normal and Fault
        # Normal: dominant frequency at 120 Hz (natural frequency)
        # Fault:  resonance shift to 98 Hz + harmonic at 196 Hz
        normal_amp = (
            0.8 * np.sin(2 * np.pi * 120 * t[:n//2])
            + 0.15 * np.sin(2 * np.pi * 240 * t[:n//2])
            + 0.05 * np.random.randn(n//2)
        )
        fault_amp = (
            1.4 * np.sin(2 * np.pi * 98 * t[n//2:])
            + 0.60 * np.sin(2 * np.pi * 196 * t[n//2:])
            + 0.12 * np.random.randn(n//2)
        )

        amplitude  = np.concatenate([normal_amp, fault_amp])
        condition  = ["Normal"] * (n // 2) + ["Fault"] * (n // 2)
        machine_id = ["Bearing_1"] * int(n * 0.6) + ["Bearing_2"] * int(n * 0.4)
        # Pad machine_id to full length
        machine_id += ["Bearing_1"] * (n - len(machine_id))

        df = pd.DataFrame({
            "time":       t,
            "amplitude":  amplitude,
            "condition":  condition,
            "machine_id": machine_id,
            "rpm":        np.random.choice([600, 800, 1000], n),
        })

        # Inject some nulls and duplicates to demonstrate cleaning
        null_idx = np.random.choice(df.index, size=200, replace=False)
        df.loc[null_idx, "amplitude"] = np.nan
        df = pd.concat([df, df.iloc[:50]], ignore_index=True)  # 50 duplicate rows

        df.to_csv(self.config["input_path"], index=False)
        self.raw_df = df
        print(f"[OK] Synthetic dataset created: {df.shape[0]} rows")

    # =========================================================================
    # MODULE 2 — DATA CLEANING
    # =========================================================================

    def clean(self) -> None:
        """
        Automated data cleaning pipeline:
          - Missing/null value detection and removal
          - Duplicate record removal
          - Data type correction
          - Unique programmatic filter (no sharing rule)
          - Export cleaned CSV
        """
        print("\n" + "="*60)
        print("MODULE 2 — DATA CLEANING")
        print("="*60)

        try:
            df = self.raw_df.copy()
            initial_rows = len(df)

            # ── Step 1: Missing value detection
            null_counts = df.isnull().sum()
            total_nulls = null_counts.sum()
            print(f"\nMissing values detected:")
            print(null_counts[null_counts > 0].to_string() if total_nulls > 0
                  else "  None found.")
            df.dropna(inplace=True)
            after_null = len(df)
            print(f"  Rows after null removal: {after_null} (removed {initial_rows - after_null})")

            # ── Step 2: Duplicate removal
            before_dup = len(df)
            df.drop_duplicates(inplace=True)
            df.reset_index(drop=True, inplace=True)
            after_dup = len(df)
            print(f"\nDuplicate rows removed: {before_dup - after_dup}")

            # ── Step 3: Data type correction
            print("\nCorrecting data types...")
            for col in df.columns:
                if df[col].dtype == object:
                    try:
                        df[col] = pd.to_numeric(df[col])
                        print(f"  Converted '{col}' → numeric")
                    except (ValueError, TypeError):
                        pass  # keep as categorical string

            # ── Step 4: Unique programmatic filter (prevents duplicate submissions)
            filt_col = self.config["unique_filter_col"]
            filt_val = self.config["unique_filter_val"]

            if filt_col in df.columns:
                before_filter = len(df)
                df = df[df[filt_col] == filt_val].copy()
                df.reset_index(drop=True, inplace=True)
                print(f"\nUnique filter applied: {filt_col} == '{filt_val}'")
                print(f"  Rows after filter: {len(df)} (from {before_filter})")
            else:
                print(f"\n[WARN] Filter column '{filt_col}' not found — skipping filter.")

            # ── Step 5: Amplitude column validation
            amp_col = self.config["amplitude_col"]
            if amp_col not in df.columns:
                raise KeyError(
                    f"Amplitude column '{amp_col}' not found. "
                    f"Available: {list(df.columns)}"
                )

            # Remove physical outliers (beyond ±5 standard deviations)
            arr = df[amp_col].to_numpy(dtype=float)
            mean, std = np.mean(arr), np.std(arr)
            mask = np.abs(arr - mean) <= 5 * std
            removed_outliers = (~mask).sum()
            df = df[mask].reset_index(drop=True)
            print(f"\nOutliers removed (>5σ): {removed_outliers}")

            # ── Save cleaned dataset
            df.to_csv(self.config["output_clean_path"], index=False)
            self.clean_df = df

            print(f"\n[OK] Cleaning complete.")
            print(f"     Final shape : {df.shape[0]} rows × {df.shape[1]} cols")
            print(f"     Saved to    : {self.config['output_clean_path']}")

        except KeyError as e:
            print(f"[ERROR] Column error during cleaning — {e}")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Unexpected error during cleaning — {e}")
            sys.exit(1)

    # =========================================================================
    # MODULE 3 — STATISTICAL ANALYSIS
    # =========================================================================

    def analyze(self) -> None:
        """
        Engineering data analytics using NumPy:
          - Descriptive statistics (mean, median, std, variance)
          - Distribution analysis (skewness, kurtosis, IQR, outliers)
          - FFT frequency domain analysis (natural frequency extraction)
          - Correlation analysis (amplitude vs time)
          - Comparative analysis (Normal vs Fault conditions)
        """
        print("\n" + "="*60)
        print("MODULE 3 — STATISTICAL ANALYSIS")
        print("="*60)

        try:
            df      = self.clean_df
            amp_col = self.config["amplitude_col"]
            arr     = df[amp_col].to_numpy(dtype=float)

            # ── 3.1 Descriptive Statistics (NumPy — mandatory)
            print("\n── 3.1 Descriptive Statistics (NumPy)")
            self.stats["mean"]     = np.mean(arr)
            self.stats["median"]   = np.median(arr)
            self.stats["std"]      = np.std(arr)
            self.stats["variance"] = np.var(arr)
            self.stats["min"]      = np.min(arr)
            self.stats["max"]      = np.max(arr)
            self.stats["range"]    = np.ptp(arr)   # peak-to-peak = max - min

            for k, v in self.stats.items():
                print(f"  {k:<12}: {v:.6f}")

            # ── 3.2 Distribution Analysis
            print("\n── 3.2 Distribution Analysis")
            self.stats["skewness"]  = float(skew(arr))
            self.stats["kurtosis"]  = float(kurtosis(arr))
            q1  = np.percentile(arr, 25)
            q3  = np.percentile(arr, 75)
            iqr = q3 - q1
            self.stats["Q1"]  = q1
            self.stats["Q3"]  = q3
            self.stats["IQR"] = iqr
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            outlier_mask = (arr < lower_fence) | (arr > upper_fence)
            self.stats["outlier_count"] = int(np.sum(outlier_mask))
            self.stats["outlier_pct"]   = 100.0 * np.mean(outlier_mask)

            print(f"  Skewness     : {self.stats['skewness']:.4f}  "
                  f"({'right-skewed' if self.stats['skewness']>0 else 'left-skewed'})")
            print(f"  Kurtosis     : {self.stats['kurtosis']:.4f}  "
                  f"({'leptokurtic' if self.stats['kurtosis']>0 else 'platykurtic'})")
            print(f"  IQR          : {iqr:.6f}")
            print(f"  Outliers     : {self.stats['outlier_count']} "
                  f"({self.stats['outlier_pct']:.2f}% of data)")

            # ── 3.3 FFT — Natural Frequency Extraction
            print("\n── 3.3 FFT Natural Frequency Analysis")
            fs = self.config["sample_rate_hz"]
            N  = len(arr)

            yf   = np.abs(fft(arr))[:N // 2]   # one-sided magnitude spectrum
            xf   = fftfreq(N, d=1.0 / fs)[:N // 2]  # frequency axis (Hz)

            # Normalize magnitude
            yf_norm = yf / N

            # Find dominant peaks
            peaks, props = find_peaks(yf_norm, height=np.percentile(yf_norm, 95),
                                      distance=int(fs / 200))

            self.fft_result = {"freq": xf, "mag": yf_norm, "peaks": peaks}

            if len(peaks) > 0:
                peak_freqs = xf[peaks]
                peak_mags  = yf_norm[peaks]
                sorted_idx = np.argsort(peak_mags)[::-1]
                top_peaks  = list(zip(peak_freqs[sorted_idx][:5],
                                      peak_mags[sorted_idx][:5]))

                self.stats["natural_freq_hz"]     = float(top_peaks[0][0])
                self.stats["dominant_peak_mag"]   = float(top_peaks[0][1])
                self.stats["top_5_peaks_hz"]      = [round(f, 2) for f, _ in top_peaks]

                print(f"  Dominant natural frequency : {self.stats['natural_freq_hz']:.2f} Hz")
                print(f"  Top 5 resonance peaks (Hz) : {self.stats['top_5_peaks_hz']}")
            else:
                print("  [WARN] No significant peaks detected in FFT spectrum.")
                self.stats["natural_freq_hz"] = float(xf[np.argmax(yf_norm)])

            # ── 3.4 Correlation Analysis
                print("\n── 3.4 Correlation Analysis")
                time_col = self.config["time_col"]
                if time_col in df.columns:
                    t_arr = pd.to_datetime(df[time_col]).astype(np.int64)
                    corr, p_val = pearsonr(t_arr, arr)
                    self.stats["corr_time_amplitude"]   = float(corr)
                    self.stats["corr_p_value"]          = float(p_val)
                    corr_matrix = np.corrcoef(t_arr, arr)
                    print(f"  Pearson r (time vs amplitude) : {corr:.4f}")
                    print(f"  p-value                       : {p_val:.4e}")
                    print(f"  Correlation matrix:\n{corr_matrix}")
                else:
                    print(f"  [WARN] Time column '{time_col}' not found — skipping correlation.")

            # ── 3.5 Comparative Analysis (Normal vs Fault)
            print("\n── 3.5 Comparative Analysis — Normal vs Fault")
            grp_col = self.config["group_col"]
            if grp_col in df.columns:
                groups = df[grp_col].unique()
                self.stats["group_comparison"] = {}
                for g in groups:
                    g_arr = df.loc[df[grp_col] == g, amp_col].to_numpy(dtype=float)
                    self.stats["group_comparison"][str(g)] = {
                        "n":      len(g_arr),
                        "mean":   float(np.mean(g_arr)),
                        "std":    float(np.std(g_arr)),
                        "max":    float(np.max(g_arr)),
                    }
                    print(f"  [{g}]  n={len(g_arr)}  "
                          f"mean={np.mean(g_arr):.4f}  "
                          f"std={np.std(g_arr):.4f}  "
                          f"max={np.max(g_arr):.4f}")
            else:
                print(f"  [WARN] Group column '{grp_col}' not found — skipping comparison.")

            print(f"\n[OK] Analysis complete. {len(self.stats)} metrics computed.")

        except Exception as e:
            print(f"[ERROR] Analysis failed — {e}")
            raise

    # =========================================================================
    # MODULE 4A — STATIC VISUALIZATIONS (3 required)
    # =========================================================================

    def visualize_static(self) -> None:
        """
        Generate 3 static charts:
          1. Amplitude Distribution Histogram with normal curve overlay
          2. Boxplot comparing Normal vs Fault conditions
          3. FFT Frequency Spectrum with resonance peak annotations
        """
        print("\n" + "="*60)
        print("MODULE 4A — STATIC VISUALIZATIONS")
        print("="*60)

        df      = self.clean_df
        amp_col = self.config["amplitude_col"]
        arr     = df[amp_col].to_numpy(dtype=float)
        out     = self.config["output_dir"]

        plt.style.use("seaborn-v0_8-whitegrid")
        colors = {"Normal": "#1A73E8", "Fault": "#E8340A", "neutral": "#444444"}

        # ── CHART 1: Amplitude Distribution Histogram
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            counts, bins, _ = ax.hist(arr, bins=80, density=True,
                                      color="#1A73E8", alpha=0.65,
                                      edgecolor="white", linewidth=0.4,
                                      label="Amplitude distribution")

            # Overlay normal distribution curve
            x_range = np.linspace(arr.min(), arr.max(), 500)
            mu, sigma = np.mean(arr), np.std(arr)
            pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
            ax.plot(x_range, pdf, color="#E8340A", linewidth=2,
                    label=f"Normal curve (μ={mu:.3f}, σ={sigma:.3f})")

            # Mark mean and ±1σ
            ax.axvline(mu,          color="#E8340A", linestyle="--", alpha=0.8, linewidth=1.2)
            ax.axvline(mu + sigma,  color="#FFA500", linestyle=":",  alpha=0.7, linewidth=1.0)
            ax.axvline(mu - sigma,  color="#FFA500", linestyle=":",  alpha=0.7, linewidth=1.0)

            ax.set_xlabel("Vibration Amplitude (g)", fontsize=12)
            ax.set_ylabel("Probability Density",     fontsize=12)
            ax.set_title("Vibration Amplitude Distribution — VIB-01 (Bearing_1)",
                         fontsize=13, fontweight="bold", pad=12)
            ax.legend(fontsize=10)
            skewness_note = (f"Skewness: {self.stats.get('skewness', 0):.3f}  |  "
                             f"Kurtosis: {self.stats.get('kurtosis', 0):.3f}")
            ax.text(0.98, 0.97, skewness_note, transform=ax.transAxes,
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
            fig.tight_layout()
            path1 = os.path.join(out, "chart1_histogram.png")
            fig.savefig(path1, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] Chart 1 saved: {path1}")

        except Exception as e:
            print(f"[ERROR] Chart 1 failed — {e}")

        # ── CHART 2: Boxplot — Normal vs Fault comparison
        try:
            grp_col = self.config["group_col"]
            fig, ax = plt.subplots(figsize=(8, 6))

            if grp_col in df.columns:
                groups     = df[grp_col].unique()
                group_data = [df.loc[df[grp_col] == g, amp_col].to_numpy(dtype=float)
                              for g in groups]
                bp = ax.boxplot(group_data, labels=groups, patch_artist=True,
                                medianprops=dict(color="white", linewidth=2))
                box_colors = ["#1A73E8", "#E8340A", "#2CA02C", "#FF7F0E"]
                for patch, color in zip(bp["boxes"], box_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            else:
                ax.boxplot(arr, patch_artist=True,
                           medianprops=dict(color="white", linewidth=2),
                           boxprops=dict(facecolor="#1A73E8", alpha=0.7))

            ax.set_xlabel("Operating Condition", fontsize=12)
            ax.set_ylabel("Vibration Amplitude (g)", fontsize=12)
            ax.set_title("Amplitude Boxplot — Comparative Analysis (Normal vs Fault)",
                         fontsize=13, fontweight="bold", pad=12)

            # Annotate IQR
            ax.text(0.02, 0.98,
                    f"IQR (overall): {self.stats.get('IQR', 0):.4f}\n"
                    f"Outliers: {self.stats.get('outlier_count', 0)} "
                    f"({self.stats.get('outlier_pct', 0):.1f}%)",
                    transform=ax.transAxes, ha="left", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

            fig.tight_layout()
            path2 = os.path.join(out, "chart2_boxplot.png")
            fig.savefig(path2, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] Chart 2 saved: {path2}")

        except Exception as e:
            print(f"[ERROR] Chart 2 failed — {e}")

        # ── CHART 3: FFT Frequency Spectrum
        try:
            fig, ax = plt.subplots(figsize=(12, 5))

            xf   = self.fft_result.get("freq", np.array([]))
            yf   = self.fft_result.get("mag",  np.array([]))
            peaks = self.fft_result.get("peaks", np.array([]))

            if len(xf) > 0:
                # Plot only 0–500 Hz for readability
                mask = xf <= 500
                ax.fill_between(xf[mask], yf[mask], alpha=0.25, color="#1A73E8")
                ax.plot(xf[mask], yf[mask], color="#1A73E8", linewidth=1.0,
                        label="Frequency spectrum")

                # Mark resonance peaks
                for pk in peaks:
                    if xf[pk] <= 500:
                        ax.annotate(f"{xf[pk]:.1f} Hz",
                                    xy=(xf[pk], yf[pk]),
                                    xytext=(xf[pk] + 5, yf[pk] * 1.08),
                                    fontsize=8.5, color="#E8340A",
                                    arrowprops=dict(arrowstyle="->",
                                                    color="#E8340A",
                                                    lw=1.2))
                        ax.axvline(xf[pk], color="#E8340A", linestyle="--",
                                   alpha=0.4, linewidth=0.9)

            ax.set_xlabel("Frequency (Hz)", fontsize=12)
            ax.set_ylabel("Normalized Magnitude", fontsize=12)
            ax.set_title("FFT Frequency Spectrum — Natural Frequency Peak Detection",
                         fontsize=13, fontweight="bold", pad=12)
            fn = self.stats.get("natural_freq_hz", "N/A")
            ax.text(0.98, 0.97,
                    f"Dominant natural freq: {fn:.2f} Hz" if isinstance(fn, float) else "",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
            ax.legend(fontsize=10)
            fig.tight_layout()
            path3 = os.path.join(out, "chart3_fft_spectrum.png")
            fig.savefig(path3, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] Chart 3 saved: {path3}")

        except Exception as e:
            print(f"[ERROR] Chart 3 failed — {e}")

    # =========================================================================
    # MODULE 4B — ANIMATED VISUALIZATIONS (2 required)
    # =========================================================================

    def visualize_animated(self) -> None:
        """
        Generate 2 animated visualizations:
          1. [Matplotlib] Rolling time-domain waveform — shows vibration over time
          2. [Plotly]     Frequency spectrum build-up animation — shows FFT evolution
        """
        print("\n" + "="*60)
        print("MODULE 4B — ANIMATED VISUALIZATIONS")
        print("="*60)

        df      = self.clean_df
        amp_col = self.config["amplitude_col"]
        arr     = df[amp_col].to_numpy(dtype=float)
        out     = self.config["output_dir"]

        # ── ANIMATION 1: Matplotlib — Rolling Waveform (time-domain)
        try:
            print("[INFO] Generating Animation 1: Rolling waveform (Matplotlib)...")

            # Use first 5000 points for smooth rendering
            arr_short = arr[:5000]
            window    = 500            # samples visible at once
            step      = 25             # samples advanced per frame
            frames    = (len(arr_short) - window) // step

            fig, (ax_wave, ax_stats) = plt.subplots(
                2, 1, figsize=(12, 7),
                gridspec_kw={"height_ratios": [3, 1]}
            )
            fig.suptitle("VIB-01 — Real-Time Vibration Waveform Monitor",
                         fontsize=13, fontweight="bold")
            fig.patch.set_facecolor("#F7F9FC")

            # Waveform axis
            line,       = ax_wave.plot([], [], color="#1A73E8", linewidth=0.9)
            mean_line   = ax_wave.axhline(np.mean(arr_short), color="#E8340A",
                                          linestyle="--", linewidth=1.2,
                                          label=f"Mean = {np.mean(arr_short):.4f}")
            std_upper   = ax_wave.axhline(np.mean(arr_short) + np.std(arr_short),
                                          color="#FFA500", linestyle=":",
                                          linewidth=1.0, label="±1σ")
            std_lower   = ax_wave.axhline(np.mean(arr_short) - np.std(arr_short),
                                          color="#FFA500", linestyle=":", linewidth=1.0)
            ax_wave.set_xlim(0, window)
            ax_wave.set_ylim(arr_short.min() * 1.2, arr_short.max() * 1.2)
            ax_wave.set_ylabel("Amplitude (g)", fontsize=11)
            ax_wave.legend(fontsize=9, loc="upper right")
            ax_wave.set_facecolor("#FAFBFD")
            frame_text = ax_wave.text(0.01, 0.96, "", transform=ax_wave.transAxes,
                                      fontsize=9, va="top")

            # Stats axis (live bar)
            stat_labels = ["Mean", "Std Dev", "Max", "Min"]
            stat_vals   = [np.mean(arr_short), np.std(arr_short),
                           arr_short.max(), arr_short.min()]
            bars        = ax_stats.barh(stat_labels, stat_vals,
                                        color=["#1A73E8", "#E8340A", "#2CA02C", "#FFA500"],
                                        alpha=0.75)
            ax_stats.set_xlabel("Value (g)", fontsize=10)
            ax_stats.set_title("Live Window Statistics", fontsize=10)
            ax_stats.set_facecolor("#FAFBFD")
            fig.tight_layout(rect=[0, 0, 1, 0.96])

            def init_anim():
                line.set_data([], [])
                return line,

            def update_anim(frame):
                start = frame * step
                end   = start + window
                chunk = arr_short[start:end]
                line.set_data(range(len(chunk)), chunk)
                frame_text.set_text(f"Sample {start}–{end}  |  "
                                    f"RMS = {np.sqrt(np.mean(chunk**2)):.4f} g")
                # Update live stats bars
                w_mean = np.mean(chunk)
                w_std  = np.std(chunk)
                w_max  = np.max(chunk)
                w_min  = np.min(chunk)
                for bar, val in zip(bars, [w_mean, w_std, w_max, w_min]):
                    bar.set_width(val)
                return line, frame_text, *bars

            ani1 = animation.FuncAnimation(
                fig, update_anim, frames=frames,
                init_func=init_anim, interval=40, blit=True
            )

            path_gif = os.path.join(out, "anim1_waveform.gif")
            path_mp4 = os.path.join(out, "anim1_waveform.mp4")

            ani1.save(path_gif, writer="pillow", fps=25, dpi=100)
            print(f"[OK] Animation 1 saved: {path_gif}")

            try:
                ani1.save(path_mp4, writer="ffmpeg", fps=25, dpi=100)
                print(f"[OK] Animation 1 also saved as MP4: {path_mp4}")
            except Exception:
                print("[INFO] ffmpeg not available — GIF saved only.")

            plt.close(fig)

        except Exception as e:
            print(f"[ERROR] Animation 1 failed — {e}")

        # ── ANIMATION 2: Plotly — Frequency Spectrum Build-Up
        try:
            print("[INFO] Generating Animation 2: Frequency spectrum (Plotly HTML)...")

            fs  = self.config["sample_rate_hz"]
            arr_anim = arr[:10_000]  # use 10k samples for speed
            N   = len(arr_anim)

            # Build frames: each frame adds more data → watch spectrum "fill in"
            num_frames   = 20
            frame_sizes  = np.linspace(500, N, num_frames, dtype=int)
            plot_frames  = []

            for size in frame_sizes:
                chunk  = arr_anim[:size]
                yf_c   = np.abs(fft(chunk))[:size // 2] / size
                xf_c   = fftfreq(size, d=1.0 / fs)[:size // 2]
                mask_c = xf_c <= 600

                plot_frames.append(go.Frame(
                    data=[go.Scatter(
                        x=xf_c[mask_c],
                        y=yf_c[mask_c],
                        mode="lines",
                        line=dict(color="#1A73E8", width=1.5),
                        fill="tozeroy",
                        fillcolor="rgba(26,115,232,0.15)",
                        name="FFT Spectrum"
                    )],
                    name=str(size),
                    layout=go.Layout(
                        title_text=f"FFT Spectrum Build-Up — {size} samples "
                                   f"({size/fs*1000:.1f} ms of data)"
                    )
                ))

            # Initial frame
            init_size = int(frame_sizes[0])
            init_chunk = arr_anim[:init_size]
            init_yf    = np.abs(fft(init_chunk))[:init_size // 2] / init_size
            init_xf    = fftfreq(init_size, d=1.0 / fs)[:init_size // 2]
            init_mask  = init_xf <= 600

            fig2 = go.Figure(
                data=[go.Scatter(
                    x=init_xf[init_mask],
                    y=init_yf[init_mask],
                    mode="lines",
                    line=dict(color="#1A73E8", width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(26,115,232,0.15)",
                    name="FFT Spectrum"
                )],
                frames=plot_frames
            )

            # Mark dominant natural frequency
            fn_hz = self.stats.get("natural_freq_hz", None)
            if fn_hz and fn_hz <= 600:
                fig2.add_vline(x=fn_hz, line_dash="dash",
                               line_color="#E8340A", line_width=1.5,
                               annotation_text=f"fn = {fn_hz:.1f} Hz",
                               annotation_position="top right",
                               annotation_font_color="#E8340A")

            fig2.update_layout(
                title=dict(
                    text="VIB-01 — FFT Frequency Spectrum Build-Up Animation",
                    font=dict(size=14, color="#222")
                ),
                xaxis=dict(title="Frequency (Hz)", range=[0, 600],
                           gridcolor="#ECECEC"),
                yaxis=dict(title="Normalized Magnitude", gridcolor="#ECECEC"),
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(family="Arial", size=11),
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    y=1.08, x=0.5, xanchor="center",
                    buttons=[
                        dict(label="▶  Play",
                             method="animate",
                             args=[None, {"frame": {"duration": 180, "redraw": True},
                                          "fromcurrent": True,
                                          "transition": {"duration": 80}}]),
                        dict(label="⏸  Pause",
                             method="animate",
                             args=[[None], {"frame": {"duration": 0},
                                            "mode": "immediate"}])
                    ]
                )],
                sliders=[dict(
                    active=0,
                    steps=[dict(
                        args=[[f.name],
                              {"frame": {"duration": 100, "redraw": True},
                               "mode": "immediate"}],
                        label=f"{int(f.name)//1000}k",
                        method="animate"
                    ) for f in plot_frames],
                    x=0.05, len=0.9, y=-0.08,
                    currentvalue=dict(prefix="Samples: ",
                                      visible=True, xanchor="center")
                )]
            )

            path_html = os.path.join(out, "anim2_fft_buildup.html")
            fig2.write_html(path_html, include_plotlyjs="cdn")
            print(f"[OK] Animation 2 saved: {path_html}")

        except Exception as e:
            print(f"[ERROR] Animation 2 failed — {e}")

    # =========================================================================
    # MODULE 5 — SUMMARY REPORT (console + text file)
    # =========================================================================

    def print_summary(self) -> None:
        """Print a formatted engineering summary of all computed statistics."""
        print("\n" + "="*60)
        print("MODULE 5 — ENGINEERING SUMMARY REPORT")
        print("="*60)

        s = self.stats
        report_lines = [
            "VIB-01: Natural Frequency Resonance — Analysis Report",
            "="*56,
            f"  Natural Frequency (fn)  : {s.get('natural_freq_hz', 'N/A'):.2f} Hz",
            f"  Mean Amplitude          : {s.get('mean', 'N/A'):.6f} g",
            f"  Median Amplitude        : {s.get('median', 'N/A'):.6f} g",
            f"  Std Deviation           : {s.get('std', 'N/A'):.6f} g",
            f"  Variance                : {s.get('variance', 'N/A'):.6f} g²",
            f"  Skewness                : {s.get('skewness', 'N/A'):.4f}",
            f"  Kurtosis                : {s.get('kurtosis', 'N/A'):.4f}",
            f"  IQR                     : {s.get('IQR', 'N/A'):.6f} g",
            f"  Outliers (IQR method)   : {s.get('outlier_count', 'N/A')} "
            f"({s.get('outlier_pct', 0):.2f}%)",
            "-"*56,
            "ENGINEERING INTERPRETATION",
            "-"*56,
        ]

        fn = s.get("natural_freq_hz", 0)
        sk = s.get("skewness", 0)
        kt = s.get("kurtosis", 0)
        oc = s.get("outlier_pct", 0)

        interp = [
            f"  Natural freq {fn:.1f} Hz indicates the structure resonates at this "
            "frequency — external excitations near this value risk amplitude amplification.",
            f"  Skewness {sk:.3f} suggests the amplitude distribution is "
            + ("right-skewed — occasional high-amplitude spikes present." if sk > 0.5
               else "left-skewed — energy concentrated at higher amplitudes." if sk < -0.5
               else "approximately symmetric — stable vibration behavior."),
            f"  Kurtosis {kt:.3f} indicates a "
            + ("leptokurtic (heavy-tailed) distribution — impulsive events present, "
               "possible bearing defects." if kt > 3
               else "platykurtic distribution — mild, uniform vibration energy."),
            f"  Outlier rate {oc:.1f}% "
            + ("is elevated — investigate for resonance events or sensor noise."
               if oc > 2.0 else "is within normal range."),
        ]

        report_lines += interp
        report_lines.append("="*56)
        report_text = "\n".join(report_lines)

        print(report_text)

        report_path = os.path.join(self.config["output_dir"], "summary_report.txt")
        try:
            with open(report_path, "w") as f:
                f.write(report_text)
            print(f"\n[OK] Summary saved: {report_path}")
        except Exception as e:
            print(f"[WARN] Could not save report — {e}")

    # =========================================================================
    # ORCHESTRATOR — run()
    # =========================================================================

    def run(self) -> None:
        """Execute the full pipeline from ingestion to summary."""
        print("\n" + "★"*60)
        print("  VIB-01 ENGINEERING DATA SYSTEMS PIPELINE — START")
        print("★"*60)

        self.ingest()
        self.clean()
        self.analyze()
        self.visualize_static()
        self.visualize_animated()
        self.print_summary()

        print("\n" + "★"*60)
        print("  PIPELINE COMPLETE — all outputs saved to outputs/")
        print("★"*60)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    pipeline = VibrationPipeline(CONFIG)
    pipeline.run()