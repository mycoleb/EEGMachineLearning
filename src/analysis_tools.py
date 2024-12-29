import numpy as np
from scipy import signal, stats
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd
import mne
from datetime import datetime

class EEGAnalysisTools:
    """Tools for analyzing recorded EEG data"""
    
    def __init__(self, config):
        self.config = config
        self.analysis_results = {}
    
    def analyze_recording(self, recording_data):
        """Comprehensive analysis of recorded EEG data"""
        results = {
            'signal_quality': self.assess_signal_quality(recording_data['eeg']),
            'spectral_analysis': self.compute_spectral_features(recording_data['eeg']),
            'classification_metrics': self.compute_classification_metrics(
                recording_data['predictions'], 
                recording_data['true_labels']
            ),
            'artifacts': self.detect_artifacts(recording_data['eeg']),
            'connectivity': self.compute_connectivity(recording_data['eeg'])
        }
        self.analysis_results = results
        return results
    
    def assess_signal_quality(self, eeg_data):
        """Assess EEG signal quality metrics"""
        return {
            'snr': self._compute_snr(eeg_data),
            'line_noise': self._detect_line_noise(eeg_data),
            'channel_variance': np.var(eeg_data, axis=1),
            'flatline_segments': self._detect_flatlines(eeg_data)
        }
    
    def compute_spectral_features(self, eeg_data):
        """Compute spectral features for each frequency band"""
        features = {}
        for band_name, (fmin, fmax) in self.config.freq_bands.items():
            power = self._compute_band_power(eeg_data, fmin, fmax)
            features[band_name] = {
                'mean_power': np.mean(power),
                'peak_frequency': self._find_peak_frequency(eeg_data, fmin, fmax),
                'band_ratio': self._compute_band_ratio(power, eeg_data)
            }
        return features
    
    def compute_classification_metrics(self, predictions, true_labels):
        """Compute classification performance metrics"""
        return {
            'confusion_matrix': confusion_matrix(true_labels, predictions),
            'accuracy_over_time': self._compute_sliding_accuracy(
                predictions, true_labels
            ),
            'roc_curves': self._compute_roc_curves(predictions, true_labels)
        }
    
    def detect_artifacts(self, eeg_data):
        """Detect various types of artifacts in EEG data"""
        return {
            'muscle': self._detect_muscle_artifacts(eeg_data),
            'blinks': self._detect_blinks(eeg_data),
            'movement': self._detect_movement_artifacts(eeg_data)
        }
    
    def compute_connectivity(self, eeg_data):
        """Compute connectivity measures between channels"""
        return {
            'correlation': self._compute_correlation_matrix(eeg_data),
            'coherence': self._compute_coherence_matrix(eeg_data),
            'phase_lag': self._compute_phase_lag_index(eeg_data)
        }
    
    def export_results(self, filename_prefix):
        """Export analysis results in multiple formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export to CSV
        pd.DataFrame(self.analysis_results).to_csv(
            f'{filename_prefix}_analysis_{timestamp}.csv'
        )
        
        # Export to JSON
        with open(f'{filename_prefix}_analysis_{timestamp}.json', 'w') as f:
            json.dump(self.analysis_results, f)
        
        # Export to MAT file
        from scipy.io import savemat
        savemat(f'{filename_prefix}_analysis_{timestamp}.mat', 
               self.analysis_results)
        
        # Generate HTML report
        self._generate_html_report(
            f'{filename_prefix}_report_{timestamp}.html'
        )
    
    def _compute_snr(self, data):
        """Compute Signal-to-Noise Ratio"""
        signal_power = np.mean(np.square(data))
        noise_power = np.var(data)
        return 10 * np.log10(signal_power / noise_power)
    
    def _detect_line_noise(self, data):
        """Detect power line interference"""
        f, pxx = signal.welch(data, fs=self.config.processing['sampling_rate'])
        line_freq_mask = (f >= 49) & (f <= 51)  # For 50 Hz
        return np.max(pxx[:, line_freq_mask])
    
    def _compute_band_power(self, data, fmin, fmax):
        """Compute power in specific frequency band"""
        f, pxx = signal.welch(data, fs=self.config.processing['sampling_rate'])
        freq_mask = (f >= fmin) & (f <= fmax)
        return np.mean(pxx[:, freq_mask], axis=1)
    
    def _generate_html_report(self, filename):
        """Generate interactive HTML report with plots"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create interactive plots
        fig = make_subplots(rows=3, cols=2)
        
        # Add plots here...
        
        # Save as HTML
        fig.write_html(filename)

    def plot_analysis_results(self):
        """Plot comprehensive analysis results"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig = plt.figure(figsize=(15, 10))
        # Add visualization code here...
        plt.show()