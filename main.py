# VIB-01: Natural Frequency Resonance
# Course: Computer Programming | AY 2026
# Student: De la Pena | TUPM250620
# Pillar 10: Vibration & Noise Control

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis, pearsonr
import warnings

warnings.filterwarnings("ignore")

# --- file paths and column settings ---
INPUT_FILE    = "data/no_fault.csv"
CLEANED_FILE  = "data/dataset_cleaned.csv"
OUTPUT_FOLDER = "outputs/"
SAMPLE_RATE   = 10000       # Hz
AMP_COL       = "sensor1"
TIME_COL      = "time_x"
GROUP_COL     = "gear_fault_desc"
FILTER_COL    = "gear_fault_desc"
FILTER_VAL    = "No fault"  # my unique filter


class VibrationPipeline:
    # this class handles the full pipeline from loading to visualization

    def __init__(self):
        self.raw_data   = None
        self.clean_data = None
        self.stats      = {}
        self.fft_freq   = None
        self.fft_mag    = None
        self.fft_peaks  = None

        # make sure output folder exists
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        os.makedirs("data", exist_ok=True)

    # MODULE 1: Load the dataset
    def ingest(self):
        print("\n--- MODULE 1: DATA INGESTION ---")

        try:
            self.raw_data = pd.read_csv(INPUT_FILE)
            print(f"Loaded: {INPUT_FILE}")
            print(f"Shape: {self.raw_data.shape}")
            print(f"Columns: {list(self.raw_data.columns)}")
            print(f"Dtypes:\n{self.raw_data.dtypes}")

        except FileNotFoundError:
            print(f"Error: File not found -> {INPUT_FILE}")
            print("Please download the dataset from Kaggle and put it in data/")

        except Exception as e:
            print(f"Something went wrong while loading: {e}")

    # MODULE 2: Clean the dataset
    def clean(self):
        print("\n--- MODULE 2: DATA CLEANING ---")

        try:
            df = self.raw_data.copy()

            # check for missing values
            nulls = df.isnull().sum()
            if nulls.sum() > 0:
                print(f"Missing values found:\n{nulls[nulls > 0]}")
                df.dropna(inplace=True)
            else:
                print("No missing values found.")

            # remove duplicate rows
            before = len(df)
            df.drop_duplicates(inplace=True)
            df.reset_index(drop=True, inplace=True)
            print(f"Duplicates removed: {before - len(df)}")

            # fix data types where possible
            for col in df.columns:
                if df[col].dtype == object:
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except:
                        pass  # leave it as string if it cant convert

            # apply my unique filter (no fault gear only)
            before_filter = len(df)
            df = df[df[FILTER_COL] == FILTER_VAL].copy()
            df.reset_index(drop=True, inplace=True)
            print(f"Filter applied: {FILTER_COL} == '{FILTER_VAL}'")
            print(f"Rows kept: {len(df)} out of {before_filter}")

            # remove extreme outliers using 5 standard deviations
            arr = df[AMP_COL].to_numpy(dtype=float)
            mean = np.mean(arr)
            std  = np.std(arr)
            mask = np.abs(arr - mean) <= 5 * std
            removed = (~mask).sum()
            df = df[mask].reset_index(drop=True)
            print(f"Outliers removed (beyond 5 std): {removed}")

            # save cleaned file
            df.to_csv(CLEANED_FILE, index=False)
            self.clean_data = df
            print(f"Cleaned data saved to: {CLEANED_FILE}")
            print(f"Final shape: {df.shape}")

        except KeyError as e:
            print(f"Column not found: {e}")

        except Exception as e:
            print(f"Error during cleaning: {e}")

    # MODULE 3: Compute statistics
    def analyze(self):
        print("\n--- MODULE 3: STATISTICAL ANALYSIS ---")

        try:
            df  = self.clean_data
            arr = df[AMP_COL].to_numpy(dtype=float)

            # 3.1 basic descriptive stats using numpy (required)
            print("\n[3.1] Descriptive Statistics")
            self.stats["mean"]     = np.mean(arr)
            self.stats["median"]   = np.median(arr)
            self.stats["std"]      = np.std(arr)
            self.stats["variance"] = np.var(arr)
            self.stats["min"]      = np.min(arr)
            self.stats["max"]      = np.max(arr)
            self.stats["range"]    = np.ptp(arr)

            for k, v in self.stats.items():
                print(f"  {k}: {v:.6f}")

            # 3.2 distribution shape
            print("\n[3.2] Distribution Analysis")
            self.stats["skewness"] = float(skew(arr))
            self.stats["kurtosis"] = float(kurtosis(arr))

            q1  = np.percentile(arr, 25)
            q3  = np.percentile(arr, 75)
            iqr = q3 - q1
            self.stats["IQR"] = iqr

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = ((arr < lower) | (arr > upper)).sum()
            self.stats["outlier_count"] = int(outliers)
            self.stats["outlier_pct"]   = round(100 * outliers / len(arr), 2)

            print(f"  Skewness: {self.stats['skewness']:.4f}")
            print(f"  Kurtosis: {self.stats['kurtosis']:.4f}")
            print(f"  IQR: {iqr:.6f}")
            print(f"  Outliers: {outliers} ({self.stats['outlier_pct']}%)")

            # 3.3 FFT to find natural frequency
            print("\n[3.3] FFT - Natural Frequency Detection")
            N  = len(arr)
            fs = SAMPLE_RATE

            yf = np.abs(fft(arr))[:N // 2] / N
            xf = fftfreq(N, d=1.0 / fs)[:N // 2]

            # find the biggest peaks
            peaks, _ = find_peaks(yf, height=np.percentile(yf, 95), distance=fs // 200)

            self.fft_freq  = xf
            self.fft_mag   = yf
            self.fft_peaks = peaks

            if len(peaks) > 0:
                sorted_peaks = sorted(zip(xf[peaks], yf[peaks]),
                                      key=lambda x: x[1], reverse=True)
                self.stats["natural_freq"] = round(float(sorted_peaks[0][0]), 2)
                top5 = [round(float(f), 2) for f, _ in sorted_peaks[:5]]
                self.stats["top_peaks_hz"] = top5
                print(f"  Dominant natural frequency: {self.stats['natural_freq']} Hz")
                print(f"  Top 5 peaks: {top5}")
            else:
                self.stats["natural_freq"] = round(float(xf[np.argmax(yf)]), 2)
                print(f"  Natural frequency: {self.stats['natural_freq']} Hz")

            # 3.4 correlation between time and amplitude
            print("\n[3.4] Correlation Analysis")
            if TIME_COL in df.columns:
                try:
                    # convert datetime string to numeric for correlation
                    t_arr = pd.to_datetime(df[TIME_COL]).astype(np.int64)
                    corr, pval = pearsonr(t_arr, arr)
                    self.stats["correlation"] = round(float(corr), 4)
                    self.stats["p_value"]     = float(pval)

                    corr_matrix = np.corrcoef(t_arr, arr)
                    print(f"  Pearson r (time vs amplitude): {corr:.4f}")
                    print(f"  p-value: {pval:.4e}")
                    print(f"  Correlation matrix:\n{corr_matrix}")
                except Exception as e:
                    print(f"  Could not compute correlation: {e}")
            else:
                print(f"  Time column not found, skipping correlation.")

            # 3.5 compare groups (normal vs fault)
            print("\n[3.5] Comparative Analysis")
            if GROUP_COL in df.columns:
                self.stats["groups"] = {}
                for g in df[GROUP_COL].unique():
                    g_arr = df.loc[df[GROUP_COL] == g, AMP_COL].to_numpy(dtype=float)
                    self.stats["groups"][str(g)] = {
                        "count": len(g_arr),
                        "mean":  round(float(np.mean(g_arr)), 4),
                        "std":   round(float(np.std(g_arr)), 4),
                        "max":   round(float(np.max(g_arr)), 4),
                    }
                    print(f"  {g}: n={len(g_arr)}, "
                          f"mean={np.mean(g_arr):.4f}, "
                          f"std={np.std(g_arr):.4f}, "
                          f"max={np.max(g_arr):.4f}")

            print(f"\nAnalysis done. {len(self.stats)} metrics computed.")

        except Exception as e:
            print(f"Error during analysis: {e}")
            raise

    # MODULE 4A: Static charts (3 required)
    def visualize_static(self):
        print("\n--- MODULE 4A: STATIC VISUALIZATIONS ---")

        df  = self.clean_data
        arr = df[AMP_COL].to_numpy(dtype=float)

        plt.style.use("seaborn-v0_8-whitegrid")

        # chart 1: histogram with normal curve
        try:
            fig, ax = plt.subplots(figsize=(10, 5))

            ax.hist(arr, bins=80, density=True, color="#1A73E8",
                    alpha=0.6, edgecolor="white", label="Amplitude data")

            # overlay normal distribution
            x = np.linspace(arr.min(), arr.max(), 500)
            mu, sigma = np.mean(arr), np.std(arr)
            normal_curve = (1 / (sigma * np.sqrt(2 * np.pi))) * \
                           np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            ax.plot(x, normal_curve, color="red", linewidth=2,
                    label=f"Normal curve (mean={mu:.3f}, std={sigma:.3f})")

            ax.axvline(mu, color="red", linestyle="--", alpha=0.7)
            ax.axvline(mu + sigma, color="orange", linestyle=":", alpha=0.6)
            ax.axvline(mu - sigma, color="orange", linestyle=":", alpha=0.6)

            ax.set_xlabel("Amplitude (g)")
            ax.set_ylabel("Probability Density")
            ax.set_title("Vibration Amplitude Distribution — No Fault Gear (sensor1)")
            ax.legend()

            fig.tight_layout()
            fig.savefig(OUTPUT_FOLDER + "chart1_histogram.png", dpi=150)
            plt.close(fig)
            print("Chart 1 saved: chart1_histogram.png")

        except Exception as e:
            print(f"Chart 1 failed: {e}")

        # chart 2: boxplot
        try:
            fig, ax = plt.subplots(figsize=(7, 6))

            if GROUP_COL in df.columns:
                groups     = df[GROUP_COL].unique()
                group_data = [df.loc[df[GROUP_COL] == g, AMP_COL].values
                              for g in groups]
                bp = ax.boxplot(group_data, labels=groups, patch_artist=True)
                colors = ["#1A73E8", "#E8340A", "#2CA02C", "#FF7F0E"]
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.65)
            else:
                ax.boxplot(arr, patch_artist=True)

            ax.set_xlabel("Gear Condition")
            ax.set_ylabel("Amplitude (g)")
            ax.set_title("Amplitude Boxplot by Gear Condition")

            fig.tight_layout()
            fig.savefig(OUTPUT_FOLDER + "chart2_boxplot.png", dpi=150)
            plt.close(fig)
            print("Chart 2 saved: chart2_boxplot.png")

        except Exception as e:
            print(f"Chart 2 failed: {e}")

        # chart 3: FFT frequency spectrum
        try:
            fig, ax = plt.subplots(figsize=(12, 5))

            xf = self.fft_freq
            yf = self.fft_mag

            if xf is not None:
                # only show up to 600 Hz
                mask = xf <= 600
                ax.fill_between(xf[mask], yf[mask], alpha=0.2, color="#1A73E8")
                ax.plot(xf[mask], yf[mask], color="#1A73E8",
                        linewidth=1.0, label="FFT spectrum")

                # annotate peaks
                if self.fft_peaks is not None:
                    for pk in self.fft_peaks:
                        if xf[pk] <= 600:
                            ax.annotate(
                                f"{xf[pk]:.1f} Hz",
                                xy=(xf[pk], yf[pk]),
                                xytext=(xf[pk] + 5, yf[pk] * 1.1),
                                fontsize=8, color="red",
                                arrowprops=dict(arrowstyle="->", color="red")
                            )

            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude")
            ax.set_title("FFT Frequency Spectrum — Natural Frequency Peak Detection")
            ax.legend()

            fig.tight_layout()
            fig.savefig(OUTPUT_FOLDER + "chart3_fft_spectrum.png", dpi=150)
            plt.close(fig)
            print("Chart 3 saved: chart3_fft_spectrum.png")

        except Exception as e:
            print(f"Chart 3 failed: {e}")

    # MODULE 4B: Animated charts (2 required)
    def visualize_animated(self):
        print("\n--- MODULE 4B: ANIMATED VISUALIZATIONS ---")

        df  = self.clean_data
        arr = df[AMP_COL].to_numpy(dtype=float)

        # animation 1: rolling waveform using matplotlib
        try:
            print("Making Animation 1: Rolling waveform...")

            data   = arr[:5000]
            window = 500
            step   = 25
            frames = (len(data) - window) // step

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                           gridspec_kw={"height_ratios": [3, 1]})
            fig.suptitle("VIB-01 — Live Vibration Waveform (No Fault Gear)")

            line, = ax1.plot([], [], color="#1A73E8", linewidth=0.9)
            ax1.axhline(np.mean(data), color="red", linestyle="--",
                        linewidth=1, label=f"Mean = {np.mean(data):.4f}")
            ax1.axhline(np.mean(data) + np.std(data), color="orange",
                        linestyle=":", linewidth=0.8, label="±1 std")
            ax1.axhline(np.mean(data) - np.std(data), color="orange",
                        linestyle=":", linewidth=0.8)
            ax1.set_xlim(0, window)
            ax1.set_ylim(data.min() * 1.2, data.max() * 1.2)
            ax1.set_ylabel("Amplitude (g)")
            ax1.legend(fontsize=8)
            info_text = ax1.text(0.01, 0.95, "", transform=ax1.transAxes,
                                 fontsize=9, va="top")

            # live stats bar
            labels = ["Mean", "Std", "Max", "Min"]
            vals   = [np.mean(data), np.std(data), data.max(), data.min()]
            bars   = ax2.barh(labels, vals,
                              color=["#1A73E8", "#E8340A", "#2CA02C", "#FF7F0E"],
                              alpha=0.7)
            ax2.set_xlabel("Value (g)")
            ax2.set_title("Window Statistics")
            fig.tight_layout()

            def init():
                line.set_data([], [])
                return line,

            def update(frame):
                start = frame * step
                end   = start + window
                chunk = data[start:end]
                line.set_data(range(len(chunk)), chunk)
                rms = np.sqrt(np.mean(chunk ** 2))
                info_text.set_text(f"Samples {start}-{end}  |  RMS = {rms:.4f} g")

                for bar, val in zip(bars, [np.mean(chunk), np.std(chunk),
                                           np.max(chunk), np.min(chunk)]):
                    bar.set_width(val)
                return line, info_text, *bars

            ani = animation.FuncAnimation(fig, update, frames=frames,
                                          init_func=init, interval=40, blit=True)
            ani.save(OUTPUT_FOLDER + "anim1_waveform.gif",
                     writer="pillow", fps=25, dpi=100)
            plt.close(fig)
            print("Animation 1 saved: anim1_waveform.gif")

        except Exception as e:
            print(f"Animation 1 failed: {e}")

        # animation 2: FFT spectrum build-up using plotly
        try:
            print("Making Animation 2: FFT spectrum build-up (Plotly)...")

            data_anim = arr[:10000]
            N         = len(data_anim)
            num_frames = 20
            sizes      = np.linspace(500, N, num_frames, dtype=int)
            plot_frames = []

            for size in sizes:
                chunk = data_anim[:size]
                yf    = np.abs(fft(chunk))[:size // 2] / size
                xf    = fftfreq(size, d=1.0 / SAMPLE_RATE)[:size // 2]
                mask  = xf <= 600

                plot_frames.append(go.Frame(
                    data=[go.Scatter(
                        x=xf[mask], y=yf[mask],
                        mode="lines",
                        line=dict(color="#1A73E8", width=1.5),
                        fill="tozeroy",
                        fillcolor="rgba(26,115,232,0.15)"
                    )],
                    name=str(size),
                    layout=go.Layout(
                        title_text=f"FFT Build-Up — {size} samples"
                    )
                ))

            # first frame
            init_size  = int(sizes[0])
            init_chunk = data_anim[:init_size]
            init_yf    = np.abs(fft(init_chunk))[:init_size // 2] / init_size
            init_xf    = fftfreq(init_size, d=1.0 / SAMPLE_RATE)[:init_size // 2]
            init_mask  = init_xf <= 600

            fig2 = go.Figure(
                data=[go.Scatter(
                    x=init_xf[init_mask], y=init_yf[init_mask],
                    mode="lines",
                    line=dict(color="#1A73E8", width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(26,115,232,0.15)"
                )],
                frames=plot_frames
            )

            fn = self.stats.get("natural_freq", None)
            if fn and fn <= 600:
                fig2.add_vline(x=fn, line_dash="dash", line_color="red",
                               annotation_text=f"fn = {fn} Hz",
                               annotation_font_color="red")

            fig2.update_layout(
                title="VIB-01 — FFT Frequency Spectrum Build-Up",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Magnitude",
                plot_bgcolor="white",
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    y=1.1, x=0.5, xanchor="center",
                    buttons=[
                        dict(label="Play",
                             method="animate",
                             args=[None, {"frame": {"duration": 180,
                                                    "redraw": True},
                                          "fromcurrent": True}]),
                        dict(label="Pause",
                             method="animate",
                             args=[[None], {"frame": {"duration": 0},
                                            "mode": "immediate"}])
                    ]
                )],
                sliders=[dict(
                    steps=[dict(
                        args=[[f.name], {"frame": {"duration": 100,
                                                   "redraw": True},
                                         "mode": "immediate"}],
                        label=str(int(f.name)),
                        method="animate"
                    ) for f in plot_frames],
                    x=0.05, len=0.9, y=-0.1,
                    currentvalue=dict(prefix="Samples: ", visible=True)
                )]
            )

            fig2.write_html(OUTPUT_FOLDER + "anim2_fft_buildup.html",
                            include_plotlyjs="cdn")
            print("Animation 2 saved: anim2_fft_buildup.html")

        except Exception as e:
            print(f"Animation 2 failed: {e}")

    # MODULE 5: Print summary
    def print_summary(self):
        print("\n--- MODULE 5: SUMMARY REPORT ---")

        s = self.stats
        fn  = s.get("natural_freq", "N/A")
        sk  = s.get("skewness", 0)
        kt  = s.get("kurtosis", 0)
        opc = s.get("outlier_pct", 0)

        lines = [
            "",
            "VIB-01 Natural Frequency Resonance — Results",
            "=" * 50,
            f"  Natural Frequency     : {fn} Hz",
            f"  Mean Amplitude        : {s.get('mean', 0):.6f} g",
            f"  Median Amplitude      : {s.get('median', 0):.6f} g",
            f"  Std Deviation         : {s.get('std', 0):.6f} g",
            f"  Variance              : {s.get('variance', 0):.6f} g^2",
            f"  Skewness              : {sk:.4f}",
            f"  Kurtosis              : {kt:.4f}",
            f"  IQR                   : {s.get('IQR', 0):.6f} g",
            f"  Outliers (IQR)        : {s.get('outlier_count', 0)} ({opc}%)",
            "-" * 50,
            "Engineering Interpretation:",
            "-" * 50,
        ]

        if isinstance(fn, float):
            lines.append(f"  - Natural frequency of {fn} Hz means the gear system")
            lines.append("    resonates at this frequency under normal conditions.")
            lines.append("    Excitations near this value could amplify vibration.")

        if sk > 0.5:
            lines.append("  - Positive skew: occasional high amplitude spikes present.")
        elif sk < -0.5:
            lines.append("  - Negative skew: energy concentrated at higher amplitudes.")
        else:
            lines.append("  - Near-symmetric distribution: stable vibration pattern.")

        if kt > 3:
            lines.append("  - Leptokurtic (heavy tails): impulsive events detected,")
            lines.append("    could indicate early-stage gear wear or resonance bursts.")
        else:
            lines.append("  - Platykurtic: mild and uniform vibration energy.")

        if opc > 2.0:
            lines.append(f"  - Outlier rate of {opc}% is high — worth investigating")
            lines.append("    for resonance events or sensor noise.")
        else:
            lines.append(f"  - Outlier rate of {opc}% is within acceptable range.")

        lines.append("=" * 50)

        report = "\n".join(lines)
        print(report)

        try:
            with open(OUTPUT_FOLDER + "summary_report.txt", "w") as f:
                f.write(report)
            print("\nReport saved: summary_report.txt")
        except Exception as e:
            print(f"Could not save report: {e}")

   
    def run(self):
        print("\n" + "=" * 50)
        print("  VIB-01 PIPELINE STARTING...")
        print("=" * 50)

        self.ingest()
        self.clean()
        self.analyze()
        self.visualize_static()
        self.visualize_animated()
        self.print_summary()

        print("\n" + "=" * 50)
        print("  DONE! All outputs saved to outputs/")
        print("=" * 50)



if __name__ == "__main__":
    pipeline = VibrationPipeline()
    pipeline.run()