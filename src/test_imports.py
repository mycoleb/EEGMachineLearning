def test_imports():
    """Test all required imports and print their versions."""
    import sys
    print(f"Python version: {sys.version}")
    
    try:
        import mne
        print(f"MNE version: {mne.__version__}")
    except ImportError as e:
        print(f"Failed to import MNE: {e}")
    
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"Failed to import NumPy: {e}")
    
    try:
        import matplotlib
        print(f"Matplotlib version: {matplotlib.__version__}")
    except ImportError as e:
        print(f"Failed to import Matplotlib: {e}")
    
    try:
        from sklearn import __version__ as sklearn_version
        print(f"Scikit-learn version: {sklearn_version}")
    except ImportError as e:
        print(f"Failed to import Scikit-learn: {e}")
    
    try:
        import joblib
        print(f"Joblib version: {joblib.__version__}")
    except ImportError as e:
        print(f"Failed to import Joblib: {e}")
    
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import classification_report
        print("All scikit-learn modules imported successfully")
    except ImportError as e:
        print(f"Failed to import some scikit-learn modules: {e}")
    
    try:
        import json
        print("JSON module imported successfully")
    except ImportError as e:
        print(f"Failed to import JSON: {e}")
        
    try:
        import logging
        print("Logging module imported successfully")
    except ImportError as e:
        print(f"Failed to import Logging: {e}")

if __name__ == "__main__":
    print("Testing imports...")
    test_imports()
    print("\nDone testing imports.")