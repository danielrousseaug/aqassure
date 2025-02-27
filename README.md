# AQ Assure: CNN for Air Quality Prediction

## Overview
AQ Assure is a convolutional neural network (CNN) model designed to predict air quality by analyzing satellite images and weather data. The model aims to provide accurate PM2.5 level predictions, helping individuals, particularly those with respiratory conditions, plan outdoor activities based on air pollution levels.

## Motivation
Growing up with asthma in Monterrey, Mexico, where air pollution is a significant concern, inspired the development of this project. The goal is to create a predictive tool that informs users about air quality, aiding individuals in making health-conscious decisions regarding outdoor activities.

## Methodology
The project follows a structured pipeline consisting of:
1. **Data Fetching**: Gathering PM2.5 measurements from OpenAQ, weather data from OpenWeather API, and planned integration of satellite imagery (currently replaced with randomized noise due to computational limitations).
2. **Data Preparation**: Aligning and preprocessing raw data for training by synchronizing datasets, normalizing satellite images, and structuring feature arrays.
3. **Model Definition**: A CNN processes satellite images, extracting spatial features, while fully connected layers handle weather data, combining both for air quality prediction.
4. **Training & Evaluation**: Training the model using PyTorch with the Adam optimizer and Mean Squared Error loss function, followed by validation.

## File Structure
```
├── config/              # Configuration files (API keys, settings)
├── data_fetching/       # Scripts for retrieving PM2.5, weather, and satellite data
├── data_preparation/    # Preprocessing and alignment of datasets
├── model.py             # Neural network architecture (CNN + Dense layers)
├── training.py          # Training and validation scripts
├── utils/               # Helper functions for data processing and visualization
└── README.md            # Project documentation
```

## Model Architecture
- **Satellite Data Processing**: A CNN extracts spatial features using convolutional layers, followed by max pooling to reduce dimensionality.
- **Weather Data Processing**: Numerical weather features (temperature, humidity, etc.) are fed into fully connected layers.
- **Fusion Layer**: The outputs from both branches are concatenated and passed through dense layers to predict PM2.5 levels.

## Sample Output
Training the model for 10 epochs outputs logs showing validation performance at each step.
```
Epoch 1/10 - Loss: 0.345
Epoch 2/10 - Loss: 0.312
...
Epoch 10/10 - Loss: 0.198
```

## Future Improvements
- **Integration of Actual Satellite Imagery**: Currently omitted due to computational limitations.
- **Advanced Architectures**: Experimenting with CNN-LSTM to capture temporal patterns in air quality data.
- **Optimization Techniques**: Exploring hyperparameter tuning and model ensembling for better performance.

## Installation & Usage
### Prerequisites
- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- OpenAQ API access
- OpenWeather API access

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/AQ-Assure.git
cd AQ-Assure

# Install dependencies
pip install -r requirements.txt
```

### Running the Model
```bash
python training.py
```

## Contributors
- **Daniel Russo** - Project Creator

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
