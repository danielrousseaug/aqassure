import numpy as np
from src.data_fetching import fetch_pm25_data, fetch_weather_data, fetch_satellite_data

def prepare_dataset(start_date, end_date):
    pm_data = fetch_pm25_data(start_date, end_date)
    pm_data = pm_data.tz_convert(None).resample('D').mean().dropna()
    pm_data.index = pm_data.index.normalize()

    weather_data = fetch_weather_data(start_date, end_date)
    if weather_data.index.tz is not None:
        weather_data.index = weather_data.index.tz_localize(None)

    common_dates = pm_data.index.intersection(weather_data.index)
    X_images, X_weather, Y = [], [], []

    for d in common_dates:
        aod_img = fetch_satellite_data(d.strftime('%Y-%m-%d'))
        aod_img = (aod_img - np.min(aod_img)) / (np.max(aod_img) - np.min(aod_img) + 1e-6)
        X_images.append(aod_img)
        X_weather.append(weather_data.loc[d].values)
        Y.append(pm_data.loc[d].pm25)

    X_images = np.array(X_images)[..., np.newaxis]
    X_weather = np.array(X_weather)
    Y = np.array(Y)
    return X_images, X_weather, Y
