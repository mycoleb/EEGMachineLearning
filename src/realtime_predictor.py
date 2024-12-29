import numpy as np
import mne
from queue import Queue
from threading import Thread
import time
from pylsl import StreamInlet, resolve_byprop
import matplotlib.pyplot as plt
import scipy.signal
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class EEGValidator:
    """Validation utilities for EEG data"""
    @staticmethod
    def validate_data_format(data):
        """Validate EEG data format"""
        if not isinstance(data, np.ndarray):
            raise ValueError("EEG data must be a numpy array")
        
        if len(data.shape) != 2:
            raise ValueError(f"Expected 2D array (channels × samples), got shape {data.shape}")
    
    @staticmethod
    def validate_data_quality(data):
        """Check EEG data quality"""
        issues = []
        
        # Check for NaN or Inf values
        if np.isnan(data).any():
            issues.append("Data contains NaN values")
        if np.isinf(data).any():
            issues.append("Data contains infinite values")
            
        # Check for flat signals
        flat_channels = np.where(np.std(data, axis=1) < 1e-6)[0]
        if len(flat_channels) > 0:
            issues.append(f"Flat signals detected in channels: {flat_channels}")
            
        # Check for extreme values
        threshold = 100  # microvolts
        extreme_values = np.abs(data) > threshold
        if extreme_values.any():
            channels = np.where(np.any(extreme_values, axis=1))[0]
            issues.append(f"Extreme values detected in channels: {channels}")
            
        return issues
    
    @staticmethod
    def validate_model(model):
        """Validate the prediction model"""
        required_methods = ['predict', 'predict_proba']
        for method in required_methods:
            if not hasattr(model, method):
                raise ValueError(f"Model must have method: {method}")

class RealTimeEEGPredictor:
    """Real-time EEG prediction system."""
    
    def __init__(self, model, buffer_size=1000, channel_names=None, sfreq=160):
        self.validator = EEGValidator()
        
        # Validate inputs
        if buffer_size <= 0:
            raise ValueError("Buffer size must be positive")
        if sfreq <= 0:
            raise ValueError("Sampling frequency must be positive")
            
        # Validate model
        self.validator.validate_model(model)
        
        self.model = model
        self.buffer_size = buffer_size
        self.channel_names = channel_names or [f'Ch{i+1}' for i in range(64)]
        self.sfreq = sfreq
        
        self.data_buffer = Queue(maxsize=buffer_size)
        self.running = False
        
        # Initialize visualization
        plt.ion()
        self.setup_visualization()
    
    def setup_visualization(self):
        """Setup visualization windows"""
        self.fig = plt.figure(figsize=(15, 10))
        gs = self.fig.add_gridspec(3, 2)
        
        self.ax_eeg = self.fig.add_subplot(gs[0, :])
        self.ax_spec = self.fig.add_subplot(gs[1, 0])
        self.ax_prob = self.fig.add_subplot(gs[1, 1])
        self.ax_quality = self.fig.add_subplot(gs[2, 0])
        self.ax_features = self.fig.add_subplot(gs[2, 1])
        
        plt.tight_layout()
    
    def preprocess_chunk(self, chunk):
        """Preprocess a chunk of EEG data with validation"""
        try:
            # Validate data format
            self.validator.validate_data_format(chunk)
            
            # Check data quality
            quality_issues = self.validator.validate_data_quality(chunk)
            if quality_issues:
                logger.warning("Data quality issues: %s", quality_issues)
            
            # Create MNE Raw object
            info = mne.create_info(
                ch_names=self.channel_names[:chunk.shape[0]],
                sfreq=self.sfreq,
                ch_types=['eeg'] * chunk.shape[0]
            )
            raw = mne.io.RawArray(chunk.T, info)
            
            # Apply preprocessing
            raw.filter(8, 30)
            raw.notch_filter(freqs=[50, 60])
            
            return raw
            
        except Exception as e:
            logger.error("Error in preprocessing: %s", str(e))
            raise
    
    def start_streaming(self, visualization_callback=None):
        """Start real-time EEG processing and prediction."""
        logger.info("Looking for EEG stream...")
        try:
            streams = resolve_byprop('type', 'EEG')
            
            if not streams:
                logger.warning("No EEG stream found, starting simulation mode")
                self.start_simulation()
                return
                
            inlet = StreamInlet(streams[0])
            self.running = True
            
            self._process_stream(inlet, visualization_callback)
            
        except Exception as e:
            logger.error("Error in streaming: %s", str(e))
            self.stop_streaming()
            raise
    
    def _process_stream(self, inlet, visualization_callback):
        """Process the EEG stream with error handling"""
        while self.running:
            try:
                chunk, timestamp = inlet.pull_chunk()
                if chunk:
                    chunk = np.array(chunk)
                    
                    # Validate and preprocess
                    raw = self.preprocess_chunk(chunk)
                    features = self.extract_features(raw)
                    
                    # Make prediction
                    prediction = self.model.predict(features)[0]
                    probabilities = self.model.predict_proba(features)
                    
                    # Update visualization
                    self.update_visualization(chunk, prediction, probabilities)
                    
                    # Store in buffer
                    if self.data_buffer.full():
                        self.data_buffer.get()
                    self.data_buffer.put((chunk, prediction, probabilities))
                    
                    if visualization_callback:
                        visualization_callback(chunk, prediction, probabilities)
                
                time.sleep(0.001)
                
            except Exception as e:
                logger.error("Error processing chunk: %s", str(e))
                continue
    
    def update_visualization(self, chunk, prediction, probabilities):
        """Update visualization with error handling"""
        try:
            # Clear previous plots
            self.ax_eeg.clear()
            self.ax_spec.clear()
            self.ax_prob.clear()
            self.ax_quality.clear()
            self.ax_features.clear()
            
            # Plot EEG signals
            times = np.arange(chunk.shape[1]) / self.sfreq
            for i, channel in enumerate(chunk):
                self.ax_eeg.plot(times, channel + i*200)
            self.ax_eeg.set_title('Real-time EEG Signal')
            
            # Plot spectrogram
            f, t, Sxx = scipy.signal.spectrogram(chunk[0], fs=self.sfreq)
            self.ax_spec.pcolormesh(t, f, 10 * np.log10(Sxx))
            self.ax_spec.set_title('Spectrogram')
            
            # Plot probabilities
            self.ax_prob.bar(range(len(probabilities[0])), probabilities[0])
            self.ax_prob.set_title(f'Predicted Class: {prediction}')
            
            # Plot data quality
            quality_issues = self.validator.validate_data_quality(chunk)
            self.ax_quality.text(0.1, 0.5, '\n'.join(quality_issues or ['No issues']),
                               transform=self.ax_quality.transAxes)
            self.ax_quality.set_title('Data Quality')
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)
            
        except Exception as e:
            logger.error("Error in visualization: %s", str(e))
            
    def start_simulation(self):
        """Start simulation mode with synthetic data"""
        logger.info("Starting simulation mode")
        self.running = True
        
        def simulation_processor():
            while self.running:
                try:
                    # Generate synthetic EEG data
                    chunk = np.random.randn(len(self.channel_names), 100) * 10
                    
                    # Add some artificial artifacts
                    if np.random.random() < 0.1:  # 10% chance of artifacts
                        chunk[0] = 100  # Extreme value
                    
                    raw = self.preprocess_chunk(chunk)
                    features = self.extract_features(raw)
                    
                    prediction = self.model.predict(features)[0]
                    probabilities = self.model.predict_proba(features)
                    
                    self.update_visualization(chunk, prediction, probabilities)
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error("Error in simulation: %s", str(e))
                    continue
        
        self.processor_thread = Thread(target=simulation_processor)
        self.processor_thread.start()
    
    def stop_streaming(self):
        """Stop streaming and cleanup"""
        logger.info("Stopping EEG stream")
        self.running = False
        if hasattr(self, 'processor_thread'):
            self.processor_thread.join()
        plt.close(self.fig)
        
    def __del__(self):
        """Cleanup on deletion"""
        self.stop_streaming()    def update_histories(self, prediction, probabilities, features, current_time):
        """Update history arrays"""
        self.prediction_history.append(prediction)
        self.probability_history.append(probabilities[0])
        self.time_history.append(current_time)
        if features is not None:
            self.feature_history.append(features)
            
        # Maintain history length
        if len(self.prediction_history) > self.config.history_length:
            self.prediction_history.pop(0)
            self.probability_history.pop(0)
            self.time_history.pop(0)
            if self.feature_history:
                self.feature_history.pop(0)

    def plot_eeg(self, chunk):
        """Plot EEG signals with enhanced visualization"""
        times = np.arange(chunk.shape[1]) / self.sfreq
        for i, channel in enumerate(chunk):
            self.ax_eeg.plot(times, channel + i*200, 
                           label=f'Channel {i+1}',
                           color=self.config.colors['lines'][i % len(self.config.colors['lines'])])
        
        self.ax_eeg.set_title('Real-time EEG Signal')
        self.ax_eeg.set_xlabel('Time (s)')
        self.ax_eeg.set_ylabel('Channel')
        self.ax_eeg.legend(loc='right')
        self.ax_eeg.grid(True, color=self.config.colors['grid'])

    def plot_spectrogram(self, chunk):
        """Plot enhanced spectrogram"""
        for i, channel in enumerate(chunk):
            f, t, Sxx = scipy.signal.spectrogram(
                channel, 
                fs=self.sfreq,
                nperseg=self.config.spectrogram_params['nperseg'],
                noverlap=self.config.spectrogram_params['noverlap']
            )
            
            mask = (f >= self.config.spectrogram_params['fmin']) & \
                  (f <= self.config.spectrogram_params['fmax'])
            
            self.ax_spec.pcolormesh(t, f[mask], 
                                  10 * np.log10(Sxx[mask]),
                                  shading='gouraud',
                                  cmap='viridis')
            
        self.ax_spec.set_title('Spectrogram')
        self.ax_spec.set_xlabel('Time (s)')
        self.ax_spec.set_ylabel('Frequency (Hz)')

    def plot_probabilities(self, probabilities, prediction):
        """Plot current prediction probabilities"""
        bar_positions = range(len(probabilities[0]))
        bars = self.ax_prob.bar(bar_positions, probabilities[0],
                               color=[self.config.colors['lines'][i % len(self.config.colors['lines'])]
                                     for i in range(len(probabilities[0]))])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            self.ax_prob.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}',
                            ha='center', va='bottom')
        
        self.ax_prob.set_title(f'Current Prediction: Class {prediction}')
        self.ax_prob.set_xlabel('Class')
        self.ax_prob.set_ylabel('Probability')

    def start_streaming(self, visualization_callback=None):
        """Start real-time EEG processing and prediction"""
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
                    features = self.extract_features(raw)
                    
                    # Make prediction
                    prediction = self.model.predict(features)[0]
                    probabilities = self.model.predict_proba(features)
                    
                    # Update visualization
                    if not self.update_visualization(chunk, prediction, 
                                                   probabilities, features):
                        break
                    
                    if visualization_callback:
                        visualization_callback(chunk, prediction, probabilities)
                
                time.sleep(0.001)
        
        self.processor_thread = Thread(target=stream_processor)
        self.processor_thread.start()

    def stop_streaming(self):
        """Stop the real-time processing"""
        self.running = False
        if hasattr(self, 'processor_thread'):
            self.processor_thread.join()
        plt.close(self.fig)
        
        # Export any remaining data if recording
        if self.recording and self.export_data:
            self.export_visualization(None)