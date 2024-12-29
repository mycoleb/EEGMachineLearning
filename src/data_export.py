import numpy as np
import pandas as pd
import json
import mne
from scipy.io import savemat
import h5py
from datetime import datetime
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import base64
import os
import zipfile

import numpy as np
import pandas as pd
import json
import mne
from scipy.io import savemat
import h5py
from datetime import datetime
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import base64
import os
import zipfile
from schema import Schema, And, Or, Optional
import plotly.express as px
from plotly.subplots import make_subplots

class EEGDataExporter:    
    def __init__(self, config):
        self.config = config
        self.supported_formats = [
            'csv', 'json', 'mat', 'edf', 'h5', 'png', 'svg', 
            'html', 'pdf', 'xlsx'
        ]
    
    def export_data(self, data, base_filename):
        """Export data in all configured formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = f"exports/{timestamp}"
        os.makedirs(export_path, exist_ok=True)
        
        results = {}
        for format in self.config.export_config['formats']:
            if format in self.supported_formats:
                try:
                    filepath = f"{export_path}/{base_filename}.{format}"
                    method = getattr(self, f"_export_to_{format}")
                    method(data, filepath)
                    results[format] = filepath
                except Exception as e:
                    results[format] = f"Export failed: {str(e)}"
        
        # Create compressed archive if configured
        if self.config.export_config['compression']:
            self._compress_exports(export_path, timestamp)
        
        return results
    
    def _export_to_csv(self, data, filepath):
        """Export to CSV format"""
        # Convert data to pandas DataFrame
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    def _export_to_json(self, data, filepath):
        """Export to JSON format"""
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def _export_to_mat(self, data, filepath):
        """Export to MATLAB format"""
        savemat(filepath, {'eeg_data': data})
    
    def _export_to_edf(self, data, filepath):
        """Export to EDF format"""
        info = mne.create_info(
            ch_names=self.config.channel_names,
            sfreq=self.config.processing['sampling_rate'],
            ch_types=['eeg'] * len(self.config.channel_names)
        )
        raw = mne.io.RawArray(data['eeg'], info)
        raw.export(filepath, fmt='edf')
    
    def _export_to_h5(self, data, filepath):
        """Export to HDF5 format"""
        with h5py.File(filepath, 'w') as f:
            for key, value in data.items():
                f.create_dataset(key, data=value)
    
    def _export_to_html(self, data, filepath):
        """Export to interactive HTML report"""
        fig = go.Figure()
        # Add interactive plots here
        pio.write_html(fig, filepath)
    
    def _export_to_png(self, data, filepath):
        """Export visualizations to PNG"""
        self._create_summary_plot(data)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _export_to_svg(self, data, filepath):
        """Export visualizations to SVG"""
        self._create_summary_plot(data)
        plt.savefig(filepath, format='svg', bbox_inches='tight')
        plt.close()
    
    def _create_summary_plot(self, data):
        """Create comprehensive summary plot"""
        plt.figure(figsize=(15, 10))
        # Add plotting code here
    
    def _create_summary_plot(self, data):
        """Create comprehensive summary plot"""
        plt.figure(figsize=(15, 10))
        
        # Create subplot grid
        gs = plt.GridSpec(3, 2)
        
        # 1. Raw EEG Plot
        ax1 = plt.subplot(gs[0, :])
        for i, channel in enumerate(data['eeg']):
            ax1.plot(channel + i*200, label=f'Channel {i+1}')
        ax1.set_title('Raw EEG Signal')
        ax1.set_xlabel('Time (samples)')
        ax1.set_ylabel('Channel')
        ax1.legend()
        
        # 2. Spectrogram
        ax2 = plt.subplot(gs[1, 0])
        if 'spectrogram' in data:
            ax2.pcolormesh(data['spectrogram'], shading='gouraud')
            ax2.set_title('Spectrogram')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Frequency (Hz)')
        
        # 3. Classification Results
        ax3 = plt.subplot(gs[1, 1])
        if 'predictions' in data:
            unique_classes = np.unique(data['predictions'])
            ax3.hist(data['predictions'], bins=len(unique_classes))
            ax3.set_title('Prediction Distribution')
            ax3.set_xlabel('Class')
            ax3.set_ylabel('Count')
        
        # 4. Feature Importance
        ax4 = plt.subplot(gs[2, 0])
        if 'feature_importance' in data:
            importance = data['feature_importance']
            ax4.bar(range(len(importance)), importance)
            ax4.set_title('Feature Importance')
            ax4.set_xlabel('Feature Index')
            ax4.set_ylabel('Importance')
        
        # 5. Performance Metrics
        ax5 = plt.subplot(gs[2, 1])
        if 'accuracy' in data:
            ax5.plot(data['accuracy'], label='Accuracy')
            ax5.set_title('Performance Over Time')
            ax5.set_xlabel('Time')
            ax5.set_ylabel('Accuracy')
        
        plt.tight_layout()

    def _compress_exports(self, export_path, timestamp):
        """Compress exported files into a zip archive"""
        zip_path = f"{export_path}/../eeg_export_{timestamp}.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all files in the export directory
            for root, _, files in os.walk(export_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, export_path)
                    zipf.write(file_path, arcname)
            
            # Add metadata if configured
            if self.config.export_config['include_metadata']:
                metadata = {
                    'timestamp': timestamp,
                    'config': self.config.to_dict(),
                    'formats': list(self.supported_formats),
                    'compression_type': 'ZIP_DEFLATED'
                }
                
                # Write metadata to a JSON file in the zip
                zipf.writestr('metadata.json', json.dumps(metadata, indent=2))
        
        return zip_path
    
    def _export_to_pdf(self, data, filepath):
        """Export to PDF format with summary plots and metadata"""
        from matplotlib.backends.backend_pdf import PdfPages
        
        with PdfPages(filepath) as pdf:
            # Create summary plot
            self._create_summary_plot(data)
            pdf.savefig()
            plt.close()
            
            # Add metadata page
            fig = plt.figure(figsize=(8, 11))
            plt.axis('off')
            metadata_text = (
                f"EEG Data Export\n"
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Channels: {len(data['eeg'])}\n"
                f"Duration: {len(data['eeg'][0])/self.config.processing['sampling_rate']:.2f} seconds\n"
                f"Sampling Rate: {self.config.processing['sampling_rate']} Hz"
            )
            plt.text(0.1, 0.9, metadata_text, transform=fig.transFigure, 
                    fontsize=12, va='top')
            pdf.savefig()
            plt.close()

    def _export_to_xlsx(self, data, filepath):
        """Export to Excel format with multiple sheets"""
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # EEG data
            pd.DataFrame(data['eeg'].T).to_excel(writer, sheet_name='EEG_Data', 
                                               index=False)
            
            # Predictions if available
            if 'predictions' in data:
                pd.DataFrame({
                    'Predictions': data['predictions'],
                    'Probabilities': data.get('probabilities', [])
                }).to_excel(writer, sheet_name='Predictions', index=False)
            
            # Feature importance if available
            if 'feature_importance' in data:
                pd.DataFrame({
                    'Feature': range(len(data['feature_importance'])),
                    'Importance': data['feature_importance']
                }).to_excel(writer, sheet_name='Feature_Importance', index=False)
            
            # Metadata
            pd.DataFrame([{
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Channels': len(data['eeg']),
                'Sampling_Rate': self.config.processing['sampling_rate'],
                'Duration': len(data['eeg'][0])/self.config.processing['sampling_rate']
            }]).to_excel(writer, sheet_name='Metadata', index=False)

        """Handle data export in multiple formats with compression"""
    
    def __init__(self, config):
        self.config = config
        self.supported_formats = [
            'csv', 'json', 'mat', 'edf', 'h5', 'png', 'svg', 
            'html', 'pdf', 'xlsx'
        ]
        self.validator = DataValidator(config)
        self.html_generator = HTMLReportGenerator(config)
    
    def export_data(self, data, base_filename):
        """Export data in all configured formats with validation"""
        # Validate data first
        validation_results = self.validator.validate_data(data)
        data['validation_results'] = validation_results
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = f"exports/{timestamp}"
        os.makedirs(export_path, exist_ok=True)
        
        results = {}
        
        # Generate HTML report first
        if 'html' in self.config.export_config['formats']:
            html_path = f"{export_path}/{base_filename}.html"
            self.html_generator.generate_report(data, validation_results, html_path)
            results['html'] = html_path
        
        # Export in each format
        for format in self.config.export_config['formats']:
            if format != 'html' and format in self.supported_formats:
                try:
                    filepath = f"{export_path}/{base_filename}.{format}"
                    method = getattr(self, f"_export_to_{format}")
                    method(data, filepath)
                    results[format] = filepath
                except Exception as e:
                    results[format] = f"Export failed: {str(e)}"
        
        # Create compressed archive if configured
        if self.config.export_config['compression']:
            zip_path = self._compress_exports(export_path, timestamp)
            results['zip'] = zip_path
        
        return results
    
    def _export_to_csv(self, data, filepath):
        """Export to CSV format"""
        export_dict = {
            'timestamps': data.get('timestamps', np.arange(data['eeg'].shape[1])),
        }
        
        # Add EEG channels
        for i, channel in enumerate(data['eeg']):
            export_dict[f'channel_{i+1}'] = channel
        
        # Add additional data if available
        if 'predictions' in data:
            export_dict['predictions'] = data['predictions']
        if 'probabilities' in data:
            for i, prob in enumerate(data['probabilities'].T):
                export_dict[f'probability_class_{i}'] = prob
        
        df = pd.DataFrame(export_dict)
        df.to_csv(filepath, index=False)
    
    def _export_to_json(self, data, filepath):
        """Export to JSON format"""
        json_data = {
            'eeg': data['eeg'].tolist(),
            'timestamps': data.get('timestamps', list(range(data['eeg'].shape[1]))).tolist(),
            'metadata': data.get('metadata', {})
        }
        
        # Add optional data
        if 'predictions' in data:
            json_data['predictions'] = data['predictions'].tolist()
        if 'probabilities' in data:
            json_data['probabilities'] = data['probabilities'].tolist()
        if 'feature_importance' in data:
            json_data['feature_importance'] = data['feature_importance'].tolist()
        
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    def _export_to_mat(self, data, filepath):
        """Export to MATLAB format"""
        mat_data = {
            'eeg_data': data['eeg'],
            'timestamps': data.get('timestamps', np.arange(data['eeg'].shape[1])),
            'metadata': str(data.get('metadata', {}))  # Convert to string for MATLAB
        }
        
        # Add optional data
        if 'predictions' in data:
            mat_data['predictions'] = data['predictions']
        if 'probabilities' in data:
            mat_data['probabilities'] = data['probabilities']
        
        savemat(filepath, mat_data)
    
    def _export_to_edf(self, data, filepath):
        """Export to EDF format"""
        info = mne.create_info(
            ch_names=self.config.channel_names[:data['eeg'].shape[0]],
            sfreq=self.config.processing['sampling_rate'],
            ch_types=['eeg'] * data['eeg'].shape[0]
        )
        raw = mne.io.RawArray(data['eeg'], info)
        raw.export(filepath, fmt='edf')
    
    def _export_to_h5(self, data, filepath):
        """Export to HDF5 format"""
        with h5py.File(filepath, 'w') as f:
            # Create main group for EEG data
            eeg_group = f.create_group('eeg')
            eeg_group.create_dataset('signals', data=data['eeg'])
            eeg_group.create_dataset('timestamps', 
                                   data=data.get('timestamps', 
                                               np.arange(data['eeg'].shape[1])))
            
            # Store metadata
            meta_group = f.create_group('metadata')
            for key, value in data.get('metadata', {}).items():
                meta_group.attrs[key] = value
            
            # Store predictions and probabilities if available
            if 'predictions' in data:
                f.create_dataset('predictions', data=data['predictions'])
            if 'probabilities' in data:
                f.create_dataset('probabilities', data=data['probabilities'])
    
    def _export_to_xlsx(self, data, filepath):
        """Export to Excel format with multiple sheets"""
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # EEG data
            eeg_df = pd.DataFrame(data['eeg'].T, 
                                columns=[f'Channel_{i+1}' for i in range(data['eeg'].shape[0])])
            eeg_df.to_excel(writer, sheet_name='EEG_Data', index=False)
            
            # Predictions if available
            if 'predictions' in data:
                pred_df = pd.DataFrame({
                    'Predictions': data['predictions'],
                    'Timestamps': data.get('timestamps', range(len(data['predictions'])))
                })
                if 'probabilities' in data:
                    for i in range(data['probabilities'].shape[1]):
                        pred_df[f'Probability_Class_{i}'] = data['probabilities'][:, i]
                pred_df.to_excel(writer, sheet_name='Predictions', index=False)
            
            # Metadata
            pd.DataFrame([data.get('metadata', {})]).to_excel(writer, 
                                                            sheet_name='Metadata', 
                                                            index=False)
    
    def _compress_exports(self, export_path, timestamp):
        """Compress exported files into a zip archive"""
        zip_path = f"{export_path}/../eeg_export_{timestamp}.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all files in the export directory
            for root, _, files in os.walk(export_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, export_path)
                    zipf.write(file_path, arcname)
            
            # Add metadata if configured
            if self.config.export_config['include_metadata']:
                metadata = {
                    'timestamp': timestamp,
                    'config': self.config.to_dict(),
                    'formats': list(self.supported_formats),
                    'compression_type': 'ZIP_DEFLATED'
                }
                zipf.writestr('metadata.json', json.dumps(metadata, indent=2))
        
        return zip_path
class DataValidator:
    """Validate EEG data structure and quality"""
    
    def __init__(self, config):
        self.config = config
        self.schema = self._create_schema()
        
    def _create_schema(self):
        """Create validation schema for EEG data"""
        return Schema({
            'eeg': And(np.ndarray, lambda x: len(x.shape) == 2),
            Optional('predictions'): And(np.ndarray, lambda x: len(x.shape) == 1),
            Optional('probabilities'): And(np.ndarray, lambda x: len(x.shape) == 2),
            Optional('feature_importance'): And(np.ndarray, lambda x: len(x.shape) == 1),
            Optional('timestamps'): And(np.ndarray, lambda x: len(x.shape) == 1),
            Optional('metadata'): dict
        })
    
    def validate_data(self, data):
        """Validate data structure and quality"""
        validation_results = {
            'structure_valid': True,
            'quality_checks': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            # Validate data structure
            self.schema.validate(data)
        except Exception as e:
            validation_results['structure_valid'] = False
            validation_results['errors'].append(f"Data structure validation failed: {str(e)}")
            return validation_results
        
        # Quality checks
        validation_results['quality_checks'] = {
            'signal_quality': self._check_signal_quality(data['eeg']),
            'missing_data': self._check_missing_data(data),
            'artifacts': self._check_artifacts(data['eeg']),
            'data_range': self._check_data_range(data['eeg'])
        }
        
        return validation_results
    
    def _check_signal_quality(self, eeg_data):
        """Check EEG signal quality"""
        return {
            'snr': self._calculate_snr(eeg_data),
            'flatline_segments': self._detect_flatlines(eeg_data),
            'noise_levels': self._calculate_noise_levels(eeg_data)
        }
    
    def _check_missing_data(self, data):
        """Check for missing or invalid data"""
        return {
            'nan_count': np.isnan(data['eeg']).sum(),
            'inf_count': np.isinf(data['eeg']).sum(),
            'zero_segments': self._count_zero_segments(data['eeg'])
        }
    
    def _check_artifacts(self, eeg_data):
        """Detect common EEG artifacts"""
        return {
            'muscle_artifacts': self._detect_muscle_artifacts(eeg_data),
            'blinks': self._detect_blinks(eeg_data),
            'movement': self._detect_movement(eeg_data)
        }
    
    def _check_data_range(self, eeg_data):
        """Check if data is within expected ranges"""
        return {
            'min_value': np.min(eeg_data),
            'max_value': np.max(eeg_data),
            'mean': np.mean(eeg_data),
            'std': np.std(eeg_data)
        }
    
    # Helper methods for quality checks
    def _calculate_snr(self, data):
        """Calculate Signal-to-Noise Ratio"""
        signal_power = np.mean(np.square(data))
        noise = data - np.mean(data, axis=1, keepdims=True)
        noise_power = np.mean(np.square(noise))
        return 10 * np.log10(signal_power / noise_power) if noise_power != 0 else float('inf')
    
    def _detect_flatlines(self, data, threshold=1e-6):
        """Detect segments with no variation"""
        return np.sum(np.abs(np.diff(data, axis=1)) < threshold, axis=1)
    
    def _calculate_noise_levels(self, data):
        """Calculate noise levels in different frequency bands"""
        from scipy import signal
        f, pxx = signal.welch(data, fs=self.config.processing['sampling_rate'])
        return {band: np.mean(pxx[:, (f >= fmin) & (f <= fmax)])
                for band, (fmin, fmax) in self.config.freq_bands.items()}

class HTMLReportGenerator:
    """Generate interactive HTML reports for EEG data"""
    
    def __init__(self, config):
        self.config = config
        
    def generate_report(self, data, validation_results, filepath):
        """Generate comprehensive HTML report"""
        # Create main figure with subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Raw EEG Signal',
                'Spectrogram',
                'Prediction Distribution',
                'Feature Importance',
                'Signal Quality Metrics',
                'Artifacts Detection'
            )
        )
        
        # Add interactive plots
        self._add_raw_eeg_plot(fig, data, row=1, col=1)
        self._add_spectrogram_plot(fig, data, row=1, col=2)
        self._add_predictions_plot(fig, data, row=2, col=1)
        self._add_feature_importance_plot(fig, data, row=2, col=2)
        self._add_quality_metrics_plot(fig, validation_results, row=3, col=1)
        self._add_artifacts_plot(fig, validation_results, row=3, col=2)
        
        # Update layout
        fig.update_layout(
            height=1200,
            width=1600,
            title_text="EEG Analysis Report",
            showlegend=True
        )
        
        # Add validation results summary
        summary_text = self._create_validation_summary(validation_results)
        fig.add_annotation(text=summary_text, x=0, y=1.1, showarrow=False)
        
        # Save interactive HTML
        fig.write_html(
            filepath,
            include_plotlyjs=True,
            full_html=True,
            include_mathjax=False
        )
        
        return filepath
    
    def _add_raw_eeg_plot(self, fig, data, row, col):
        """Add interactive raw EEG plot"""
        for i, channel in enumerate(data['eeg']):
            fig.add_trace(
                go.Scatter(
                    y=channel + i*200,
                    name=f'Channel {i+1}',
                    line=dict(width=1)
                ),
                row=row, col=col
            )
    
    def _add_spectrogram_plot(self, fig, data, row, col):
        """Add interactive spectrogram plot"""
        if 'spectrogram' in data:
            fig.add_trace(
                go.Heatmap(
                    z=data['spectrogram'],
                    colorscale='Viridis'
                ),
                row=row, col=col
            )
    
    def _add_predictions_plot(self, fig, data, row, col):
        """Add predictions distribution plot"""
        if 'predictions' in data:
            fig.add_trace(
                go.Histogram(x=data['predictions']),
                row=row, col=col
            )
    
    def _add_feature_importance_plot(self, fig, data, row, col):
        """Add feature importance plot"""
        if 'feature_importance' in data:
            fig.add_trace(
                go.Bar(
                    x=range(len(data['feature_importance'])),
                    y=data['feature_importance']
                ),
                row=row, col=col
            )
    
    def _add_quality_metrics_plot(self, fig, validation_results, row, col):
        """Add signal quality metrics plot"""
        metrics = validation_results['quality_checks']['signal_quality']
        fig.add_trace(
            go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values())
            ),
            row=row, col=col
        )
    
    def _add_artifacts_plot(self, fig, validation_results, row, col):
        """Add artifacts detection plot"""
        artifacts = validation_results['quality_checks']['artifacts']
        fig.add_trace(
            go.Bar(
                x=list(artifacts.keys()),
                y=list(artifacts.values())
            ),
            row=row, col=col
        )
    
    def _create_validation_summary(self, validation_results):
        """Create text summary of validation results"""
        return f"""
        Data Validation Summary:
        - Structure Valid: {validation_results['structure_valid']}
        - Warnings: {len(validation_results['warnings'])}
        - Errors: {len(validation_results['errors'])}
        """
