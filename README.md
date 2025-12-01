# seizure-detector

Deep-learning model for automated seizure detection from behavioral videos.

## Overview

This project processes multi-view cage recordings (TOP, SIDE1, SIDE2) of mice
and trains a CNN-based model to detect seizure events from behavior.

Ground-truth seizure times are currently annotated manually by an observer
watching the raw videos and entering start/end times into a spreadsheet.

## Data and alignment

We assume that the three camera views are reasonably synchronized at recording time.
Because seizure labels are obtained by visually inspecting the original videos,
we are **not** running an automatic alignment step in the core pipeline right now.

The script `src/alignment_check.py` is kept as an optional diagnostic tool:
it estimates temporal offsets between views using brightness-based
cross-correlation and writes a summary to `data/alignment_summary.csv`.
This script is not required for the main training and evaluation pipeline.
