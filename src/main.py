import argparse
from pathlib import Path
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

from classifier import EEGClassifier
from deep_models import EEGNet, DeepConvNet, train_model
from realtime_predictor import RealTimeEEGPredictor
from analysis_tools import EEGAnalysisTools
from visualizations import EEGVisualizer
from config import EEGVisualizationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """Validate input data and configurations"""
    @staticmethod
    def validate_config(config):
        required_fields = [
            'channel_names', 'processing', 'freq_bands', 
            'export_config', 'plot_config'
        ]
        for field in required_fields:
            if not hasattr(config, field):
                raise ValueError(f"Missing required config field: {field}")
        
        if not config.channel_names:
            raise ValueError("Channel names cannot be empty")
        
        if config.processing['sampling_rate'] <= 0:
            raise ValueError("Sampling rate must be positive")
    
    @staticmethod
    def validate_eeg_data(data):
        if not isinstance(data, np.ndarray):
            raise ValueError("EEG data must be a numpy array")
        
        if len(data.shape) != 2:
            raise ValueError("EEG data must be 2D (channels × samples)")
        
        if np.isnan(data).any():
            raise ValueError("EEG data contains NaN values")
        
        if np.isinf(data).any():
            raise ValueError("EEG data contains infinite values")

def prepare_deep_learning_data(epochs, config):
    """Prepare data for deep learning models"""
    X = epochs.get_data()  # (trials, channels, samples)
    y = epochs.events[:, -1]
    
    # Convert to torch tensors
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X, y)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.processing.get('batch_size', 32),
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.processing.get('batch_size', 32),
        shuffle=False
    )
    
    return train_loader, val_loader

def train_deep_model(model, train_loader, val_loader, config):
    """Train deep learning model with visualization"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training visualization setup
    import matplotlib.pyplot as plt
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    train_losses = []
    val_accuracies = []
    
    n_epochs = config.processing.get('n_epochs', 50)
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss/(batch_idx+1)})
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        # Update visualizations
        train_losses.append(total_loss/len(train_loader))
        val_accuracies.append(100 * correct / total)
        
        ax1.clear()
        ax1.plot(train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        
        ax2.clear()
        ax2.plot(val_accuracies)
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        
        plt.tight_layout()
        plt.pause(0.1)
    
    plt.ioff()
    return model

class RealTimeVisualizer:
    """Enhanced real-time visualization"""
    def __init__(self, config):
        self.config = config
        self.setup_visualization()
        
    def setup_visualization(self):
        import matplotlib.pyplot as plt
        plt.ion()
        self.fig = plt.figure(figsize=(15, 10))
        gs = self.fig.add_gridspec(3, 2)
        
        # EEG signal plot
        self.ax_eeg = self.fig.add_subplot(gs[0, :])
        # Spectrogram
        self.ax_spec = self.fig.add_subplot(gs[1, 0])
        # Prediction probabilities
        self.ax_prob = self.fig.add_subplot(gs[1, 1])
        # Topographic map
        self.ax_topo = self.fig.add_subplot(gs[2, 0])
        # Feature importance
        self.ax_feat = self.fig.add_subplot(gs[2, 1])
        
        plt.tight_layout()
    
    def update(self, eeg_data, prediction, probabilities, features=None):
        # Update EEG signal plot
        self.ax_eeg.clear()
        for i, channel in enumerate(eeg_data):
            self.ax_eeg.plot(channel + i*200)
        self.ax_eeg.set_title('Real-time EEG Signal')
        
        # Update spectrogram
        self.ax_spec.clear()
        # Add spectrogram code here...
        
        # Update probabilities
        self.ax_prob.clear()
        self.ax_prob.bar(range(len(probabilities)), probabilities)
        self.ax_prob.set_title(f'Prediction: Class {prediction}')
        
        # Update topographic map
        self.ax_topo.clear()
        # Add topographic map code here...
        
        # Update feature importance
        if features is not None:
            self.ax_feat.clear()
            self.ax_feat.bar(range(len(features)), features)
            self.ax_feat.set_title('Feature Importance')
        
        plt.tight_layout()
        plt.pause(0.01)

def parse_arguments():
    parser = argparse.ArgumentParser(description='EEG Signal Classification System')
    parser.add_argument('--mode', type=str, default='train',
                      choices=['train', 'predict', 'analyze', 'export'],
                      help='Operation mode')
    parser.add_argument('--model', type=str, default='svm',
                      choices=['svm', 'eegnet', 'deepconv'],
                      help='Model type to use')
    parser.add_argument('--subjects', type=int, nargs='+', default=[1],
                      help='Subject numbers to process')
    parser.add_argument('--visualize', action='store_true',
                      help='Enable visualization')
    parser.add_argument('--validate', action='store_true',
                      help='Enable strict data validation')
    return parser.parse_args()

def main():
    args = parse_arguments()
    config = EEGVisualizationConfig()
    
    # Validate configuration if requested
    if args.validate:
        logger.info("Validating configuration...")
        DataValidator.validate_config(config)
    
    # Create necessary directories
    Path('models').mkdir(exist_ok=True)
    Path('exports').mkdir(exist_ok=True)
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    
    if args.mode == 'train':
        logger.info(f"Training {args.model} model for subjects {args.subjects}")
        
        if args.model == 'svm':
            classifier = EEGClassifier()
            for subject in args.subjects:
                data = classifier.load_subject_data(subject)
                if args.validate:
                    DataValidator.validate_eeg_data(data)
                classifier.train_subject(subject)
                
        elif args.model in ['eegnet', 'deepconv']:
            # Deep learning pipeline
            model_class = EEGNet if args.model == 'eegnet' else DeepConvNet
            model = model_class()
            
            for subject in args.subjects:
                # Load and validate data
                data = load_subject_data(subject)
                if args.validate:
                    DataValidator.validate_eeg_data(data)
                
                # Prepare data for deep learning
                train_loader, val_loader = prepare_deep_learning_data(data, config)
                
                # Train model with visualization
                model = train_deep_model(model, train_loader, val_loader, config)
                
                # Save model
                torch.save(model.state_dict(), 
                         f'models/{args.model}_subject_{subject}.pt')
            
    elif args.mode == 'predict':
        logger.info("Starting real-time prediction...")
        
        # Load appropriate model
        if args.model == 'svm':
            classifier = EEGClassifier()
            classifier.load_model('models/eeg_classifier.joblib')
            model = classifier.model
        else:
            model_class = EEGNet if args.model == 'eegnet' else DeepConvNet
            model = model_class()
            model.load_state_dict(torch.load(f'models/{args.model}_latest.pt'))
            model.eval()
        
        # Initialize predictor and visualizer
        predictor = RealTimeEEGPredictor(
            model=model,
            channel_names=config.channel_names,
            sfreq=config.processing['sampling_rate']
        )
        
        if args.visualize:
            visualizer = RealTimeVisualizer(config)
            predictor.set_visualization_callback(visualizer.update)
        
        predictor.start_streaming()
        
    elif args.mode == 'analyze':
        logger.info("Analyzing EEG data...")
        analyzer = EEGAnalysisTools(config)
        results = analyzer.analyze_recording()
        
        if args.visualize:
            visualizer = EEGVisualizer(results['raw'], results['epochs'])
            visualizer.plot_brain_activity_map()
            visualizer.plot_connectivity_matrix()
        
        analyzer.export_results('analysis_output')
        
    elif args.mode == 'export':
        logger.info("Exporting data...")
        from data_export import EEGDataExporter
        exporter = EEGDataExporter(config)
        exporter.export_data(data, 'eeg_export')

if __name__ == "__main__":
    main()