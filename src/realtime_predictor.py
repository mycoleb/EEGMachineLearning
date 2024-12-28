import numpy as np
import mne
from queue import Queue
from threading import Thread
import time
from pylsl import StreamInlet, resolve_stream

class RealTimeEEGPredictor:
    """
    Real-time EEG prediction system.
    """
    def __init__(self, model, buffer_size=1000, channel_names=None, sfreq=160):
        self.model = model
        self.buffer_size = buffer_size
        self.channel_names = channel_names
        self.sfreq = sfreq
        
        self.data_buffer = Queue(maxsize=buffer_size)
        self.running = False
    
    def preprocess_chunk(self, chunk):
        """
        Preprocess a chunk of EEG data.
        """
        # Create MNE Raw object
        info = mne.create_info(
            ch_names=self.channel_names,
            sfreq=self.sfreq,
            ch_types=['eeg'] * len(self.channel_names)
        )
        raw = mne.io.RawArray(chunk.T, info)
        
        # Apply same preprocessing as training
        raw.filter(8, 30)
        raw.notch_filter(freqs=[50, 60])
        
        return raw
    
    def extract_features(self, raw):
        """
        Extract features from preprocessed data chunk.
        """
        data = raw.get_data()
        
        # Extract frequency band features
        freq_bands = {
            'mu': (8, 12),
            'beta': (12, 30)
        }
        
        features = []
        for channel in data:
            for band_name, (fmin, fmax) in freq_bands.items():
                psds, freqs = mne.time_frequency.psd_array_welch(
                    channel,
                    sfreq=raw.info['sfreq'],
                    fmin=fmin,
                    fmax=fmax,
                    n_fft=256
                )
                features.append(np.mean(psds))
        
        return np.array(features).reshape(1, -1)
    
    def start_streaming(self, visualization_callback=None):
        """
        Start real-time EEG processing and prediction.
        """
        print("Looking for an EEG stream...")
        streams = resolve_stream('type', 'EEG')
        inlet = StreamInlet(streams[0])
        
        self.running = True
        
        def stream_processor():
            while self.running:
                chunk, timestamp = inlet.pull_chunk()
                if chunk:
                    chunk = np.array(chunk)
                    raw = self.preprocess_chunk(chunk)
                    features =