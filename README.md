# EEG Signal Classification System

A machine learning system for classifying motor imagery EEG signals using Python, MNE, and scikit-learn.

## Project Overview

This project implements a classification system for EEG signals, specifically focusing on motor imagery tasks (e.g., imagined left-hand vs. right-hand movements). It uses MNE-Python for EEG data processing and scikit-learn for machine learning.

## Installation

1. **Clone this repository:**
   ```bash
   git clone [your-repository-url]
   cd eeg-classification
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Data Structure

Place your EEG data files in the `data/raw/` directory. The system supports EDF format files.

## Usage

1. **Prepare your data:**
   - Place your EDF files in the `data/raw/` directory.
   - Ensure your events are properly marked in the EEG recordings.

2. **Run the classifier:**
   ```bash
   python -m src.classifier
   ```

## Features

- EEG data preprocessing with bandpass filtering (8-30 Hz).
- Epoch extraction around events.
- Feature extraction from EEG signals.
- Machine learning pipeline with standardization and SVM classification.
- Performance evaluation using classification metrics.

## Configuration

The main parameters can be adjusted in `src/classifier.py`:
- Frequency band for filtering (currently 8-30 Hz).
- Epoch duration (currently -0.5 to 1.5 seconds around events).
- Classification model parameters.