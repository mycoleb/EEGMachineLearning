import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from mne.viz import plot_topomap

class EEGVisualizer:
    """
    Advanced visualization tools for EEG data analysis.
    """
    def __init__(self, raw, epochs, model=None):
        self.raw = raw
        self.epochs = epochs
        self.model = model
        
    def plot_brain_activity_map(self, event_id=None, tmin=0.0, tmax=0.2):
        """
        Plot topographic maps of brain activity.
        """
        if event_id:
            evoked = self.epochs[event_id].average()
        else:
            evoked = self.epochs.average()
        
        times = np.linspace(tmin, tmax, 5)
        fig, axes = plt.subplots(1, len(times), figsize=(15, 3))
        
        for ax, time in zip(axes, times):
            evoked.plot_topomap(times=time, axes=ax, show=False)
            ax.set_title(f'{time:.3f} s')
        
        plt.suptitle('Brain Activity Topographic Maps')
        plt.tight_layout()
        plt.show()
    
    def plot_time_frequency(self, picks=['C3..', 'Cz..', 'C4..'], fmin=4, fmax=40):
        """
        Plot time-frequency analysis.
        """
        freqs = np.arange(fmin, fmax, 2)
        n_cycles = freqs / 2.
        
        power = mne.time_frequency.tfr_morlet(self.epochs, freqs=freqs, n_cycles=n_cycles, 
                                            return_itc=False, picks=picks)
        
        power.plot_topo(baseline=(-0.5, 0), mode='zscore', title='Time-Frequency Analysis')
        plt.show()
    
    def plot_connectivity_matrix(self):
        """
        Plot connectivity matrix between EEG channels.
        """
        data = self.epochs.get_data()
        n_channels = data.shape[1]
        
        # Calculate correlation matrix
        corr_matrix = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(n_channels):
                corr = np.corrcoef(data[:, i, :].mean(axis=1), 
                                 data[:, j, :].mean(axis=1))[0, 1]
                corr_matrix[i, j] = corr
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, xticklabels=self.epochs.ch_names, 
                   yticklabels=self.epochs.ch_names, cmap='RdBu_r')
        plt.title('Channel Connectivity Matrix')
        plt.tight_layout()
        plt.show()
    
    def plot_model_performance(self, y_true, y_pred, y_score=None):
        """
        Plot comprehensive model performance metrics.
        """
        plt.figure(figsize=(15, 5))
        
        # Confusion Matrix
        plt.subplot(131)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        
        # ROC Curve
        if y_score is not None:
            plt.subplot(132)
            n_classes = len(np.unique(y_true))
            
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_score[:, i])
                plt.plot(fpr, tpr, label=f'Class {i}')
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves')
            plt.legend()
        
        # Feature Importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            plt.subplot(133)
            importances = self.model.feature_importances_
            plt.bar(range(len(importances)), importances)
            plt.title('Feature Importance')
        
        plt.tight_layout()
        plt.show()
    
    def plot_real_time_prediction(self, data_chunk, prediction, probability):
        """
        Plot real-time prediction visualization.
        """
        plt.clf()
        
        # Plot EEG signals
        plt.subplot(211)
        plt.plot(data_chunk.T)
        plt.title('Real-time EEG Signal')
        
        # Plot prediction probabilities
        plt.subplot(212)
        plt.bar(range(len(probability)), probability)
        plt.title(f'Predicted Class: {prediction} (Probability: {max(probability):.2f})')
        
        plt.tight_layout()
        plt.pause(0.1)