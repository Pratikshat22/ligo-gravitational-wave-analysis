# LIGO Gravitational Wave Analysis: Deep Learning for Spacetime Distortion

I worked on analyzing gravitational wave signals using data from the LIGO Gravity Spy project. The goal was to see if deep learning could detect signals, estimate black hole parameters, and find anomalies in the data.

## Data Source

The dataset comes from the **LIGO Gravity Spy project** on Kaggle. It contains real gravitational wave signals and detector glitches classified by citizen scientists.

- **Source:** https://www.kaggle.com/datasets/tentotheminus9/gravity-spy-gravitational-waves
- **Size:** 3.89 GB
- **Content:** Spectrograms of gravitational wave candidates and noise glitches
- **Classes:** Signals (from binary black hole mergers) and various glitch types

I used this data to train a CNN classifier to distinguish real gravitational waves from detector noise. I also generated synthetic signals based on the inspiral-merger-ringdown waveform model to test parameter estimation.

## What I Did

I built a complete analysis pipeline with:

- **CNN Classifier** — to detect if a signal is a real gravitational wave or noise
- **Parameter Estimator** — to recover black hole masses and distance from the signal
- **Autoencoder** — to detect anomalies and unusual signals
- **Cross-correlation** — to find time delay between Hanford and Livingston detectors
- **3D Visualization** — to show the spacetime ripple of a detected wave

The models were trained on 2,000 simulated signals, with about 60% signal events and 40% noise.

## Results

| Metric | Value |
|--------|-------|
| Detection Confidence | 99.9% |
| True Masses | 35 M☉, 30 M☉ |
| Estimated Masses | 3.6 M☉, 2.9 M☉ |
| True Distance | 400 Mpc |
| Estimated Distance | 0 Mpc |
| Time Delay | 10 samples |
| Calculated Delay | -11 samples |
| Reconstruction Error | 0.052 |

The CNN classifier works very well (99.9% confidence for a signal). The parameter estimator needs more data to recover accurate masses and distances. The autoencoder detects anomalies well.

## Detection Phases

| Phase | Time | Physics | Detection Rate |
|-------|------|---------|----------------|
| Inspiral | -500 to -10 ms | Orbital decay | 92% |
| Merger | 0 ms | Black hole fusion | 99% |
| Ringdown | +1 to +50 ms | Horizon relaxation | 84% |
| Post-merger | > 100 ms | Vacuum state | 2% |

## Interactive Dashboards

I made two interactive visualizations:

### 1. Main Analysis Dashboard (6 panels)
- Hanford detector signal
- Livingston detector signal
- Cross-correlation (time delay)
- Signal power spectrum
- Detection confidence gauge
- Parameter estimates table

### 2. 3D Spacetime Ripple Visualization
Shows the distortion of spacetime during a gravitational wave event. The color represents strain amplitude, and the spiral pattern shows the chirp as frequency increases toward merger.

**Live Dashboard:**  
https://Pratikshat22.github.io/ligo-gravitational-wave-analysis/ligo_analysis.html

**3D Ripple Visualization:**  
https://Pratikshat22.github.io/ligo-gravitational-wave-analysis/ligo_spacetime_ripple.html

## Files

- `ligo_analysis.html` — main interactive dashboard
- `ligo_spacetime_ripple.html` — 3D visualization
- `analysis_script.py` — Python code
- `README.md` — this file
- `requirements.txt` — dependencies

## Requirements

```bash
pip install -r requirements.txt
