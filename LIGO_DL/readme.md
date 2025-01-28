# LIGO Gravitational Waves Detection with Deep Learning

A deep learning approach to detecting gravitational waves in LIGO data, implementing neural network architectures inspired by techniques from Goodfellow's Deep Learning book.

## Overview

This project implements a deep learning pipeline for detecting gravitational wave events in LIGO data. It features two neural network architectures:
- A baseline feed-forward neural network using standard gradient descent
- An optimized version incorporating momentum and adaptive learning rate scheduling

## Features

- **Data Preprocessing**
  - LIGO data preprocessing using gwpy
  - Whitening and bandpass filtering (30-400 Hz)
  - Automated data cleaning and normalization

- **Neural Network Models**
  - Configurable feed-forward architecture
  - L2 regularization for both models
  - Momentum-based optimization (optimized model)
  - Adaptive learning rate scheduling (optimized model)

- **Visualization & Analysis**
  - Training history visualization
  - Model performance comparisons
  - Signal waveform analysis
  - Confusion matrices
  - Spectrum analysis
  - Learning rate evolution plots

- **Logging System**
  - Comprehensive metric tracking
  - Model parameter logging
  - Performance comparison tools
  - Training history storage

## Prerequisites

- Python 3.8 or higher

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
.\venv\Scripts\activate   # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
LIGO_DL/
├── src/
│   ├── preprocessing.py     # LIGO data preprocessing pipeline
│   ├── model.py            # Neural network implementations
│   ├── visualization.py    # Visualization tools
│   ├── data_logger.py      # Logging utilities
│   └── test_model.py       # Model comparison and testing
├── results/                # Saved model results
├── plots/                  # Generated visualizations
├── tests/                  # Unit tests
├── examples/               # Usage examples
├── LICENSE                 # Project license
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## Usage

### Basic Usage

```python
from src.preprocessing import LIGODataPreprocessor
from src.model import DeepGWDetector, OptimizedGWDetector

# Initialize preprocessor
preprocessor = LIGODataPreprocessor(window_size=32)

# Create dataset
X, y = preprocessor.create_dataset([gw_event_time], [non_event_times])

# Train models
model = OptimizedGWDetector(
    input_dim=X.shape[1],
    hidden_dims=[64, 32],
    initial_learning_rate=0.005,
    momentum=0.95
)

losses = model.train(X, y, epochs=50)
```

### Running Tests

```bash
python src/test_model.py
```

## Model Architecture

### Baseline Model
- Feed-forward neural network
- ReLU activation for hidden layers
- Sigmoid activation for output layer
- L2 regularization
- Standard gradient descent
- Configurable hidden layer dimensions

### Optimized Model
- Enhanced feed-forward architecture
- Momentum-based optimization
- Learning rate scheduling
- Improved L2 regularization
- Mini-batch processing

## Data Format

### Input
- LIGO strain data
- Expected sampling rate: 4096 Hz
- Window size: 32 seconds (configurable)
- Supports H1, L1, and V1 detectors

### Output
- Binary classification (event/non-event)
- Probability scores
- Performance metrics
- Visualization artifacts

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```
4. Make your changes
5. Run tests
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- LIGO Scientific Collaboration for data access
- gwpy library developers
- Ian Goodfellow et al. for deep learning insights

## Future Improvements

- GPU acceleration support
- Additional neural network architectures
- Real-time processing capabilities
- Extended detector support
- Advanced data augmentation techniques

## Contact

For questions and support, please open an issue in the GitHub repository or send an email to the authors.