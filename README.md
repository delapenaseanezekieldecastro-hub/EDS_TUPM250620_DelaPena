# EDS Final Project — VIB-01: Natural Frequency Resonance

**Course:** Computer Programming | Academic Year: 2026  
**Student:** Dela Pena | Student No.: TUPM250620  
**Pillar:** 10 — Vibration & Noise Control | **Topic:** VIB-01  

---

## Research Topic
**Natural Frequency Resonance Analysis** using vibration signals from a Mechanical Gear System under normal operating conditions. This pipeline detects resonance peaks, computes FFT-based natural frequencies, and performs statistical analysis on real sensor data.

---

## Dataset
**Name:** Mechanical Gear Vibration Dataset  
**Source:** [Kaggle — Gear Vibration Dataset](https://www.kaggle.com/datasets/hieudaotrung/gear-vibration)  
**Description:** Vibration signals of six gear types measured under various working conditions using two sensors.

### Unique Filter Applied
```python
gear_fault_desc == 'No fault'   # Only normal (healthy) gear operation
```
This ensures a unique data slice distinct from all other student submissions.

---

## Project Structure
```
EDS_TUPM250620_DelaPena/
├── main.py                  # Full Python pipeline (OOP, 5 modules)
├── requirements.txt         # Required libraries
├── README.md                # This file
├── data/
│   ├── no_fault.csv         # Original dataset (filtered slice)
│   └── dataset_cleaned.csv  # Auto-generated cleaned dataset
└── outputs/
    ├── chart1_histogram.png      # Amplitude distribution histogram
    ├── chart2_boxplot.png        # Normal vs Fault boxplot comparison
    ├── chart3_fft_spectrum.png   # FFT frequency spectrum
    ├── anim1_waveform.gif        # Animated rolling waveform
    ├── anim2_fft_buildup.html    # Animated FFT spectrum build-up (Plotly)
    └── summary_report.txt        # Engineering summary report
```

---

## Pipeline Architecture

The pipeline is built using an Object-Oriented Programming (OOP) approach with 5 distinct modules inside the `VibrationPipeline` class:

| Module | Method | Description |
|--------|--------|-------------|
| 1 | `ingest()` | Load and validate CSV dataset with error handling |
| 2 | `clean()` | Remove nulls, duplicates, outliers, apply unique filter |
| 3 | `analyze()` | NumPy statistics, FFT, correlation, comparative analysis |
| 4a | `visualize_static()` | Generate 3 static matplotlib charts |
| 4b | `visualize_animated()` | Generate 2 animated visualizations |
| 5 | `print_summary()` | Engineering interpretation report |

---

## Key Results

| Metric | Value |
|--------|-------|
| Dataset rows (raw) | 150,000 |
| Dataset rows (cleaned) | 149,683 |
| Dominant Natural Frequency | 0.27 Hz |
| Harmonic Peaks | 167 Hz, 540 Hz, 860 Hz, 1081 Hz |
| Mean Amplitude | 2.5205 g |
| Std Deviation | 0.0077 g |
| Skewness | 0.1277 (approximately symmetric) |
| Kurtosis | 4.069 (leptokurtic — impulsive events present) |
| Outlier Rate | 10.98% |

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/DelaPena/EDS_TUPM250620_DelaPena.git
cd EDS_TUPM250620_DelaPena
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Place dataset
Download `no_fault.csv` from the Kaggle link above and place it in the `data/` folder.

### 4. Run the pipeline
```bash
python main.py
```

All outputs will be saved automatically to the `outputs/` folder.

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `numpy` | Statistical computations (mean, std, variance, FFT) |
| `pandas` | Data loading, cleaning, filtering |
| `matplotlib` | Static charts + animated waveform GIF |
| `plotly` | Interactive animated FFT spectrum (HTML) |
| `scipy` | FFT, peak detection, Pearson correlation |
| `pillow` | GIF rendering for Matplotlib animation |
| `kaleido` | Plotly static image export |

---

## Engineering Interpretation

The dominant natural frequency of **0.27 Hz** represents the fundamental resonance of the gear system under healthy (no-fault) conditions. Harmonic peaks at **540 Hz** and **860 Hz** indicate mechanical excitation frequencies from gear meshing. The **leptokurtic distribution (kurtosis = 4.07)** suggests occasional impulsive vibration events even in normal operation, which serves as a baseline for fault detection. The elevated outlier rate of **10.98%** warrants further investigation for early-stage resonance risk.

---

*Submitted for Computer Programming Final Project, Academic Year 2026*
