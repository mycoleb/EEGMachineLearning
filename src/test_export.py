from data_export import EEGDataExporter
from config import EEGVisualizationConfig
import numpy as np
import pandas as pd

def create_sample_data(num_channels=64, num_samples=1000):
    """Create sample EEG data for testing"""
    # Create EEG data
    eeg_data = np.random.randn(num_channels, num_samples)
    
    # Create predictions
    predictions = np.random.randint(0, 3, num_samples)
    
    # Create feature importance
    feature_importance = np.abs(np.random.rand(20))
    
    # Create DataFrame-compatible dictionary
    data_dict = {
        'eeg': eeg_data,
        'predictions': predictions,
        'feature_importance': feature_importance,
        'timestamps': np.arange(num_samples) / 160.0,  # Assuming 160 Hz sampling rate
        'metadata': {
            'sampling_rate': 160,
            'subject': 'test_subject',
            'num_channels': num_channels,
            'duration': num_samples / 160.0
        }
    }
    
    # Create spectrogram data
    from scipy import signal
    f, t, Sxx = signal.spectrogram(eeg_data[0], fs=160)
    data_dict['spectrogram'] = Sxx
    
    # Create probabilities
    probabilities = np.random.rand(num_samples, 3)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    data_dict['probabilities'] = probabilities
    
    return data_dict

if __name__ == "__main__":
    print("Creating sample data...")
    sample_data = create_sample_data()
    
    print("Initializing configuration...")
    config = EEGVisualizationConfig()
    
    print("Creating exporter...")
    exporter = EEGDataExporter(config)
    
    print("Exporting data...")
    results = exporter.export_data(sample_data, 'test_export')
    
    print("\nExport results:")
    for format, path in results.items():
        print(f"{format}: {'Success' if 'failed' not in str(path).lower() else 'Failed'}")
        if 'failed' in str(path).lower():
            print(f"  Error: {path}")