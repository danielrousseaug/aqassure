import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.config import PM25_API_KEY, PM25_SENSOR_ID, WEATHER_API_KEY, DEFAULT_LAT, DEFAULT_LON

def fetch_pm25_data(start_date, end_date):
    headers = {'X-API-Key': PM25_API_KEY}
    measurements_url = f'https://api.openaq.org/v3/sensors/{PM25_SENSOR_ID}/measurements/daily'
    params = {
        'datetime_from': f'{start_date}T00:00:00Z',
        'datetime_to': f'{end_date}T23:59:59Z',
        'limit': 1000
    }
    response = requests.get(measurements_url, headers=headers, params=params)
    if response.status_code == 200:
        measurements = response.json()
        if 'results' in measurements and measurements['results']:
            data = [
                (res['period']['datetimeFrom']['utc'], res['value'])
                for res in measurements['results']
                if 'period' in res and 'datetimeFrom' in res['period'] and 'utc' in res['period']['datetimeFrom']
            ]
            pm_df = pd.DataFrame(data, columns=['utc_time', 'pm25'])
            pm_df['utc_time'] = pd.to_datetime(pm_df['utc_time'])
            pm_df.set_index('utc_time', inplace=True)
            return pm_df
    return pd.DataFrame(columns=['utc_time', 'pm25'])

def fetch_weather_data(start_date, end_date, lat=DEFAULT_LAT, lon=DEFAULT_LON):
    date_range = pd.date_range(start_date, end_date)
    temp = np.random.uniform(10, 30, size=len(date_range))
    humidity = np.random.uniform(30, 80, size=len(date_range))
    weather_df = pd.DataFrame({'temp': temp, 'humidity': humidity}, index=date_range)
    return weather_df

def fetch_satellite_data(date):
    return np.random.rand(256, 256)
