import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import plot_model

def plot_pm25_trend(pm_data):
    plt.figure(figsize=(10, 6))
    plt.plot(pm_data.index, pm_data['pm25'], label='PM2.5 (µg/m³)', color='b')
    plt.axhline(12, color='r', linestyle='--', label='EPA Safe Level (12 µg/m³)')
    plt.title('PM2.5 Levels Over Time')
    plt.xlabel('Date')
    plt.ylabel('PM2.5 (µg/m³)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_satellite_image(aod_data, date):
    plt.figure(figsize=(6, 6))
    plt.imshow(aod_data, cmap='viridis')
    plt.colorbar(label='Aerosol Optical Depth (AOD)')
    plt.title(f'Simulated Satellite Image for {date}')
    plt.axis('off')
    plt.show()

def plot_weather_trends(weather_data):
    plt.figure(figsize=(12, 6))
    plt.plot(weather_data.index, weather_data['temp'], label='Temperature (°C)', color='orange')
    plt.plot(weather_data.index, weather_data['humidity'], label='Humidity (%)', color='blue')
    plt.title('Weather Trends Over Time')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_pm25_vs_weather(pm_data, weather_data):
    combined = pm_data.join(weather_data)
    plt.figure(figsize=(8, 6))
    plt.scatter(combined['temp'], combined['pm25'], color='orange', alpha=0.7, label='PM2.5 vs Temperature')
    plt.scatter(combined['humidity'], combined['pm25'], color='blue', alpha=0.7, label='PM2.5 vs Humidity')
    plt.title('PM2.5 vs Weather Features')
    plt.xlabel('Weather Features')
    plt.ylabel('PM2.5 (µg/m³)')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_model_architecture(model, filename='model.png'):
    plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)
    img = plt.imread(filename)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Model Architecture')
    plt.show()

def plot_training_metrics(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mae = history.history['mae']
    val_mae = history.history['val_mae']

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(loss, label='Training Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='orange')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(mae, label='Training MAE', color='blue')
    plt.plot(val_mae, label='Validation MAE', color='orange')
    plt.title('Training vs Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, color='blue', alpha=0.7)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label='Perfect Prediction')
    plt.title('Predicted vs Actual PM2.5')
    plt.xlabel('Actual PM2.5 (µg/m³)')
    plt.ylabel('Predicted PM2.5 (µg/m³)')
    plt.legend()
    plt.grid(True)
    plt.show()
