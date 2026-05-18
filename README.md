# EDS Final Project — VIB-01: Natural Frequency Resonance

**Course:** Computer Programming | Academic Year: 2026  
**Student:** Dela Pena | Student No.: TUPM250620  
**Pillar:** 10 — Vibration & Noise Control | **Topic:** VIB-01  

---

## What this project is about

This project analyzes vibration signals from a mechanical gear system to find its natural frequency. I used FFT (Fast Fourier Transform) to detect resonance peaks and ran some statistical analysis on the sensor data. I picked the no-fault condition as my data slice so I could get a clean baseline of what normal gear behavior looks like.

---

## Dataset

**Name:** Mechanical Gear Vibration Dataset  
**Source:** [Kaggle — Gear Vibration Dataset](https://www.kaggle.com/datasets/hieudaotrung/gear-vibration)  

The dataset has vibration readings from six gear types under different conditions, measured by two sensors. I filtered it down to only healthy gear operation:

```python
gear_fault_desc == 'No fault'
```

---

## Project Structure

```
EDS_TUPM250620_DelaPena/
├── main.py                  # main pipeline
├── requirements.txt
├── README.md
├── data/
│   ├── no_fault.csv         # filtered dataset
│   └── dataset_cleaned.csv  # auto-generated after cleaning
└── outputs/
    ├── chart1_histogram.png
    ├── chart2_boxplot.png
    ├── chart3_fft_spectrum.png
    ├── anim1_waveform.gif
    ├── anim2_fft_buildup.html
    └── summary_report.txt
```

---

## How the pipeline works

I built this using OOP with one main class (`VibrationPipeline`) split into 5 modules:

1. **ingest()** — loads and validates the CSV
2. **clean()** — handles nulls, duplicates, outliers, and applies the gear filter
3. **analyze()** — FFT, statistics, correlation
4. **visualize_static() / visualize_animated()** — 3 static charts + 2 animations
5. **print_summary()** — prints the engineering summary to a text file

---

## Results

| Metric | Value |
|--------|-------|
| Raw rows | 150,000 |
| After cleaning | 149,683 |
| Dominant Natural Frequency | 0.27 Hz |
| Harmonic Peaks | 167 Hz, 540 Hz, 860 Hz, 1081 Hz |
| Mean Amplitude | 2.5205 g |
| Std Deviation | 0.0077 g |
| Skewness | 0.1277 |
| Kurtosis | 4.069 |
| Outlier Rate | 10.98% |

The dominant frequency came out at 0.27 Hz which is the fundamental resonance of the gear under normal conditions. The harmonic peaks at 540 Hz and 860 Hz are from gear meshing. Kurtosis of 4.07 means there are some impulsive spikes even in healthy operation — which is actually useful as a fault detection baseline. The outlier rate (10.98%) is a bit high so that's something worth looking into further.

---

## How to Run

```bash
git clone https://github.com/DelaPena/EDS_TUPM250620_DelaPena.git
cd EDS_TUPM250620_DelaPena
pip install -r requirements.txt
```

Download `no_fault.csv` from the Kaggle link and drop it in the `data/` folder, then:

```bash
python main.py
```

Outputs go to the `outputs/` folder automatically.

---

## Libraries Used

- `numpy` — FFT and stats
- `pandas` — loading and cleaning the data
- `matplotlib` — static charts and the waveform GIF
- `plotly` — animated FFT HTML
- `scipy` — peak detection, Pearson correlation
- `pillow` — for rendering the GIF
- `kaleido` — Plotly image export

---

*Submitted for Computer Programming Final Project, Academic Year 2026*
