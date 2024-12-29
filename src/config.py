class EEGVisualizationConfig:
    """Configuration for EEG visualization and export"""
    def __init__(self):
        # Basic EEG parameters
        self.channel_names = [f'Ch{i+1}' for i in range(64)]  # Default channel names
        self.processing = {
            'sampling_rate': 160,
            'buffer_size': 1000,
            'update_interval': 0.1
        }
        
        # Export configuration
        self.export_config = {
            'formats': ['csv', 'json', 'mat', 'edf', 'png', 'svg', 'html'],
            'compression': True,
            'include_metadata': True
        }
        
        # Frequency bands
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        # Visualization settings
        self.plot_config = {
            'colors': {
                'background': 'white',
                'grid': 'lightgray',
                'lines': ['blue', 'green', 'red', 'purple', 'orange']
            },
            'font_size': 12,
            'dpi': 100
        }
    
    def to_dict(self):
        """Convert configuration to dictionary"""
        return {
            'channel_names': self.channel_names,
            'processing': self.processing,
            'export_config': self.export_config,
            'freq_bands': self.freq_bands,
            'plot_config': self.plot_config
        }