from src.training import train_model
from src.visualization import (
    plot_pm25_trend, plot_satellite_image, plot_weather_trends,
    plot_pm25_vs_weather, visualize_model_architecture,
    plot_training_metrics, plot_predictions
)
from src.data_fetching import fetch_pm25_data, fetch_weather_data, fetch_satellite_data
from src.data_preparation import prepare_dataset
from src.model import build_model

if __name__ == "__main__":
    start_date = '2023-01-01'
    end_date = '2023-12-31'

    # Fetch PM2.5 and weather data for visualization
    pm_data = fetch_pm25_data(start_date, end_date)
    weather_data = fetch_weather_data(start_date, end_date)

    # Standardize timezone for both DataFrames
    pm_data.index = pm_data.index.tz_localize(None)  # Remove timezone from PM2.5 data
    weather_data.index = weather_data.index.tz_localize(None)  # Ensure weather data is also tz-naive

    # Visualization 1: PM2.5 Trend
    plot_pm25_trend(pm_data)

    # Visualization 2: Weather Trends
    plot_weather_trends(weather_data)

    # Visualization 3: PM2.5 vs Weather
    plot_pm25_vs_weather(pm_data, weather_data)

    # Visualization 4: Sample Satellite Image
    aod_data = fetch_satellite_data(date='2023-07-15')
    plot_satellite_image(aod_data, date='2023-07-15')

    # Prepare dataset for training
    X_images, X_weather, Y = prepare_dataset(start_date, end_date)

    # Train the model
    model = train_model(start_date, end_date)

    # Visualization 5: Model Architecture
    visualize_model_architecture(model)

    # Visualization 6: Training Metrics
    history = model.fit(
        [X_images[:int(0.8*len(Y))], X_weather[:int(0.8*len(Y))]], 
        Y[:int(0.8*len(Y))],
        validation_data=(
            [X_images[int(0.8*len(Y)):], X_weather[int(0.8*len(Y)):]], 
            Y[int(0.8*len(Y)):]
        ),
        epochs=10, batch_size=8
    )
    plot_training_metrics(history)

    # Visualization 7: Predicted vs Actual PM2.5
    y_val_pred = model.predict([X_images[int(0.8*len(Y)):], X_weather[int(0.8*len(Y)):]]).flatten()
    plot_predictions(Y[int(0.8*len(Y)):], y_val_pred)
