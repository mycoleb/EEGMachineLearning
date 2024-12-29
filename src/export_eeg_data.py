from data_export import EEGDataExporter
from config import EEGVisualizationConfig
import numpy as np

def create_sample_data():
    """Create sample EEG data for testing"""
    # Create random EEG data: 64 channels, 1000 time points
    eeg_data = np.random.randn(64, 1000)
    
    sample_data = {
        'eeg': eeg_data,
        'timestamps': np.arange(1000) / 160.0,  # Assuming 160 Hz sampling rate
        'predictions': np.random.randint(0, 3, 1000),  # Random class predictions
        'probabilities': np.random.rand(1000, 3),  # Random class probabilities
        'metadata': {
            'subject': 'test_subject',
            'sampling_rate': 160,
            'date': '2024-12-28'
        }
    }
    
    return sample_data

def main():
    # Initialize configuration
    print("Initializing configuration...")
    config = EEGVisualizationConfig()
    
    # Create EEG data exporter
    print("Creating EEG data exporter...")
    exporter = EEGDataExporter(config)
    
    # Get your EEG data (using sample data for this example)
    print("Creating sample data...")
    your_data = create_sample_data()
    
    # Export the data
    print("Exporting data...")
    results = exporter.export_data(your_data, 'eeg_export')
    
    # Print results
    print("\nExport Results:")
    for format, result in results.items():
        if 'failed' in str(result).lower():
            print(f"{format}: Failed - {result}")
        else:
            print(f"{format}: Success - {result}")

if __name__ == "__main__":
    main()