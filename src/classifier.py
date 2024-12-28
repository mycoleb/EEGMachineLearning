import os
import logging
from pathlib import Path
import mne
import numpy as np
from mne.datasets import eegbci
from mne.io import concatenate_raws
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import json

mne.set_log_level('WARNING')
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EEGClassifier:
    """
    A class for EEG signal classification using motor imagery data.
    """
    def __init__(self, config=None):
        """
        Initialize the EEG classifier with configuration parameters.
        
        Args:
            config (dict, optional): Configuration parameters for the classifier.
        """
        self.config = {
            'freq_band': (8., 30.),  # Motor imagery relevant frequencies
            'epoch_time': (-0.5, 1.5),  # Time before and after event
            'event_id': {'left_hand': 1, 'right_hand': 2},
            'test_size': 0.2,
            'random_state': 42
        }
        if config:
            self.config.update(config)
        
        self.model = None
        self.scaler = None
    
    def preprocess_data(self, raw):
        """
        Enhanced preprocessing of the EEG data.
        
        Args:
            raw (mne.io.Raw): Raw EEG data
            
        Returns:
            mne.io.Raw: Preprocessed data
        """
        # Apply bandpass filter
        raw.filter(self.config['freq_band'][0], 
                  self.config['freq_band'][1],
                  method='iir',
                  iir_params=dict(order=4, ftype='butter'))
        
        # Apply notch filter for line noise
        raw.notch_filter(freqs=[50, 60])
        
        # Apply CAR (Common Average Reference)
        raw.set_eeg_reference(ref_channels='average')
        
        # Remove bad channels if any
        raw.interpolate_bads()
        
        return raw
    
    def extract_epochs(self, raw, events):
        """
        Extract epochs from continuous EEG data.
        
        Args:
            raw (mne.io.Raw): Raw EEG data.
            events (numpy.ndarray): Events array.
            
        Returns:
            mne.Epochs: Extracted epochs.
        """
        logger.info("Extracting epochs")
        tmin, tmax = self.config['epoch_time']
        epochs = mne.Epochs(raw, events, self.config['event_id'], 
                          tmin, tmax, proj=True, 
                          baseline=(None, 0), preload=True)
        
        return epochs
    
    def prepare_features(self, epochs):
        """
        Extract features from epochs including frequency band power.
        
        Args:
            epochs (mne.Epochs): EEG epochs.
            
        Returns:
            tuple: (X, y) features and labels.
        """
        logger.info("Preparing features")
        from tqdm import tqdm
        
        # Get the data
        X = epochs.get_data()
        y = epochs.events[:, -1]
        
        # Define frequency bands of interest
        freq_bands = {
            'mu': (8, 12),    # mu rhythm
            'beta': (12, 30)  # beta band
        }
        
        X_features = []
        total_epochs = len(X)
        
        print("\nExtracting frequency features...")
        for epoch_idx, epoch in enumerate(tqdm(X, desc="Processing epochs")):
            epoch_features = []
            for channel_idx, channel in enumerate(epoch):
                for band_name, (fmin, fmax) in freq_bands.items():
                    # Calculate power spectrum density
                    psds, freqs = mne.time_frequency.psd_array_welch(
                        channel, 
                        sfreq=epochs.info['sfreq'],
                        fmin=fmin, 
                        fmax=fmax,
                        n_fft=256
                    )
                    # Add band power as feature
                    epoch_features.append(np.mean(psds))
            X_features.append(epoch_features)
        
        X_features = np.array(X_features)
        logger.info(f"Spectral features shape: {X_features.shape}")
        
        # Add temporal features
        n_epochs, n_channels, n_times = X.shape
        X_temporal = X.reshape(n_epochs, n_channels * n_times)
        logger.info(f"Temporal features shape: {X_temporal.shape}")
        
        print("\nCombining features...")
        # Combine spectral and temporal features
        X_combined = np.hstack([X_temporal, X_features])
        
        logger.info(f"Combined feature matrix shape: {X_combined.shape}")
        return X_combined, y
    
    def create_pipeline(self):
        """
        Create an advanced machine learning pipeline with feature selection
        and multiple classifiers.
        
        Returns:
            dict: Dictionary of classification pipelines
        """
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.feature_selection import SelectFromModel
        from sklearn.decomposition import PCA
        
        pipelines = {}
        
        # SVM Pipeline with feature selection
        pipelines['svm'] = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectFromModel(
                GradientBoostingClassifier(n_estimators=100, random_state=42)
            )),
            ('pca', PCA(n_components=0.95)),
            ('classifier', SVC(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            ))
        ])
        
        # Random Forest Pipeline
        pipelines['rf'] = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectFromModel(
                RandomForestClassifier(n_estimators=100, random_state=42)
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=42
            ))
        ])
        
        # Gradient Boosting Pipeline
        pipelines['gb'] = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectFromModel(
                GradientBoostingClassifier(n_estimators=100, random_state=42)
            )),
            ('classifier', GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ))
        ])
        
        return pipelines
    
    def train(self, X, y):
        """
        Train and evaluate multiple classification models.
        
        Args:
            X (numpy.ndarray): Feature matrix.
            y (numpy.ndarray): Labels.
        """
        from sklearn.model_selection import cross_val_score
        logger.info("Training models")
        
        # Get all pipelines
        pipelines = self.create_pipeline()
        
        # Evaluate each pipeline
        best_score = 0
        best_pipeline = None
        
        print("\nEvaluating different models:")
        for name, pipeline in pipelines.items():
            # Perform cross-validation
            cv_scores = cross_val_score(pipeline, X, y, cv=5, n_jobs=-1)
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            
            print(f"\n{name.upper()} Pipeline:")
            print(f"Cross-validation scores: {cv_scores}")
            print(f"Average CV score: {mean_score:.3f} (+/- {std_score * 2:.3f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_pipeline = pipeline
        
        # Train the best model on all data
        print(f"\nTraining final model (best CV score: {best_score:.3f})...")
        self.model = best_pipeline
        self.model.fit(X, y)
        
        # Final evaluation
        y_pred = self.model.predict(X)
        logger.info("\nFinal Classification Report:")
        logger.info("\n" + classification_report(y, y_pred))
    
    def predict_realtime(self, raw_chunk):
        """
        Make real-time predictions on streaming EEG data.
        
        Args:
            raw_chunk (mne.io.RawArray): A chunk of raw EEG data
            
        Returns:
            int: Predicted class label
        """
        # Preprocess the chunk
        raw_chunk = self.preprocess_data(raw_chunk)
        
        # Extract features (similar to prepare_features but for single chunk)
        X = raw_chunk.get_data()
        
        # Define frequency bands
        freq_bands = {
            'mu': (8, 12),
            'beta': (12, 30)
        }
        
        # Extract features
        features = []
        for channel in X:
            for band_name, (fmin, fmax) in freq_bands.items():
                psds, freqs = mne.time_frequency.psd_array_welch(
                    channel,
                    sfreq=raw_chunk.info['sfreq'],
                    fmin=fmin,
                    fmax=fmax,
                    n_fft=256
                )
                features.append(np.mean(psds))
        
        # Reshape features to match training data format
        X_features = np.array(features).reshape(1, -1)
        
        # Make prediction
        return self.model.predict(X_features)[0]
    
    def plot_features(self, epochs):
        """
        Plot relevant features and EEG signals.
        
        Args:
            epochs (mne.Epochs): The epoched data
        """
        import matplotlib.pyplot as plt
        
        # Plot average power spectrum
        epochs.plot_psd(picks=['C3..', 'Cz..', 'C4..'], 
                       fmin=4, fmax=40,
                       average=True)
        plt.title('Power Spectrum Density')
        
        # Plot ERP for different conditions
        epochs.plot_image(picks=['C3..', 'C4..'],
                         combine='mean')
        plt.title('Event Related Potentials')
        
        plt.show()

    def visualize_model_performance(self, X, y):
        """
        Visualize model performance metrics.
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): True labels
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        
        # Get predictions
        y_pred = self.model.predict(X)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # For Random Forest, plot feature importance
        if 'randomforest' in str(self.model).lower():
            # Get feature importances
            importances = self.model.named_steps['classifier'].feature_importances_
            
            # Plot top 20 features
            plt.figure(figsize=(12, 6))
            n_features = min(20, len(importances))
            indices = np.argsort(importances)[-n_features:]
            
            plt.title('Top 20 Most Important Features')
            plt.barh(range(n_features), importances[indices])
            plt.yticks(range(n_features), indices)
            plt.xlabel('Feature Importance')
            plt.tight_layout()
            plt.show()
        
    def visualize_model_performance(self, X, y):
        """
        Visualize model performance metrics and feature importance.
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): True labels
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title('Confusion Matrix')
        plt.show()
        
        # If using Random Forest, plot feature importance
        if 'rf' in str(self.model) or 'RandomForest' in str(self.model):
            # Get feature importance from the classifier
            classifier = self.model.named_steps['classifier']
            importances = classifier.feature_importances_
            
            # Plot top 20 most important features
            plt.figure(figsize=(12, 6))
            n_features = 20
            indices = np.argsort(importances)[-n_features:]
            plt.title('Top 20 Most Important Features')
            plt.barh(range(n_features), importances[indices])
            plt.xlabel('Feature Importance')
            plt.ylabel('Feature Index')
            plt.tight_layout()
            plt.show()

    def save_model(self, filepath):
        """
        Save the trained model and configuration.
        
        Args:
            filepath (str): Path to save the model.
        """
        if self.model is None:
            raise ValueError("No trained model to save.")
        
        # Create directory if it doesn't exist
        model_path = Path(filepath)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model using joblib
        from joblib import dump
        dump(self.model, model_path)
        
        # Save the configuration
        config_path = model_path.parent / (model_path.stem + '_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f)
            
        logger.info(f"Model saved to {filepath}")
        logger.info(f"Configuration saved to {config_path}")

    def visualize_model_performance(self, X, y):
        """
        Visualize model performance metrics.
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): True labels
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        
        # Get predictions
        y_pred = self.model.predict(X)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # For Random Forest, plot feature importance
        if 'randomforest' in str(self.model).lower():
            # Get feature importances
            importances = self.model.named_steps['classifier'].feature_importances_
            
            # Plot top 20 features
            plt.figure(figsize=(12, 6))
            n_features = min(20, len(importances))
            indices = np.argsort(importances)[-n_features:]
            
            plt.title('Top 20 Most Important Features')
            plt.barh(range(n_features), importances[indices])
            plt.yticks(range(n_features), indices)
            plt.xlabel('Feature Importance')
            plt.tight_layout()
            plt.show()
            
        # Plot training curves if available
        if hasattr(self.model, 'cv_results_'):
            plt.figure(figsize=(10, 6))
            plt.plot(self.model.cv_results_['mean_test_score'])
            plt.title('Cross-validation Performance')
            plt.xlabel('Iteration')
            plt.ylabel('Score')
            plt.show()

def main():
    """
    Main function to run the EEG classification pipeline using PhysioNet Motor Imagery dataset.
    """
    from tqdm import tqdm
    
    # Set logging level
    mne.set_log_level('WARNING')
    
    # Download and load PhysioNet data for multiple subjects
    print("Downloading sample EEG data...")
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Test on multiple subjects
    subjects = [1, 2]  # Add more subjects as needed
    all_scores = []
    
    for subject in subjects:
        print(f"\nProcessing subject {subject}")
        runs = [6, 10, 14]  # Motor imagery: hands vs feet
        raw_files = mne.datasets.eegbci.load_data(subject, runs, path=str(data_dir))
        
        # Initialize classifier
        classifier = EEGClassifier()
        
        # Load and concatenate the files
        print("\nLoading EDF files...")
        raws = []
        for file in tqdm(raw_files, desc="Loading EDF files"):
            raw = mne.io.read_raw_edf(file, preload=True)
            raws.append(raw)
        raw = mne.concatenate_raws(raws)
        
        # Convert annotations to events
        events, event_id = mne.events_from_annotations(raw)
        event_id = {'T1': 1, 'T2': 2, 'T3': 3}
        classifier.config['event_id'] = event_id
        
        # Process the data
        epochs = classifier.extract_epochs(raw, events)
        X, y = classifier.prepare_features(epochs)
        
        # Train and evaluate
        classifier.train(X, y)
        
        # Visualize performance
        classifier.visualize_model_performance(X, y)
        
        # Save the model
        classifier.save_model(f'models/eeg_classifier_subject_{subject}.joblib')
    """
    Main function to run the EEG classification pipeline using PhysioNet Motor Imagery dataset.
    """
    from tqdm import tqdm
    
    # Download and load PhysioNet data
    print("Downloading sample EEG data...")
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download example data (motor imagery: hands vs feet)
    subject = 1  # Subject number
    runs = [6, 10, 14]  # Motor imagery: hands vs feet
    raw_files = mne.datasets.eegbci.load_data(subject, runs, path=str(data_dir))
    
    # Initialize classifier
    classifier = EEGClassifier()
    
    # Load and concatenate the files
    print("\nLoading EDF files...")
    raws = []
    for file in tqdm(raw_files, desc="Loading EDF files"):
        raw = mne.io.read_raw_edf(file, preload=True)
        raws.append(raw)
    raw = mne.concatenate_raws(raws)
    
    # Print some information about the data
    print("\nData Information:")
    print(f"Number of channels: {len(raw.ch_names)}")
    print(f"Channel names: {raw.ch_names}")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Total duration: {raw.times.max():.2f} seconds")
    
    # Rename the events for clarity
    event_id = dict(hands=2, feet=3)  # PhysioNet's event codes
    
    # Update classifier config with correct event IDs
    classifier.config['event_id'] = event_id
    
    # Convert annotations to events
    print("Converting annotations to events...")
    events, event_id = mne.events_from_annotations(raw)
    
    # Update the event_id to match the PhysioNet codes
    # T1=rest, T2=left hand imagery, T3=right hand imagery
    event_id = {'T1': 1, 'T2': 2, 'T3': 3}
    classifier.config['event_id'] = event_id
    
    # Process the data
    print("Extracting epochs...")
    epochs = classifier.extract_epochs(raw, events)
    
    # Plot the data
    print("Plotting features...")
    classifier.plot_features(epochs)
    
    # Extract features and train
    X, y = classifier.prepare_features(epochs)
    
    # Train the model
    classifier.train(X, y)
    
    # Save the model
    classifier.save_model('models/eeg_classifier.joblib')

if __name__ == "__main__":
    main()