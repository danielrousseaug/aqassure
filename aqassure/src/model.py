from tensorflow.keras import layers, models
from src.config import SATELLITE_IMAGE_SHAPE, WEATHER_FEATURE_DIM

def build_model():
    image_input = layers.Input(shape=SATELLITE_IMAGE_SHAPE)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(image_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)

    weather_input = layers.Input(shape=(WEATHER_FEATURE_DIM,))
    combined = layers.Concatenate()([x, weather_input])
    combined = layers.Dense(64, activation='relu')(combined)
    combined = layers.Dense(32, activation='relu')(combined)
    output = layers.Dense(1)(combined)

    model = models.Model(inputs=[image_input, weather_input], outputs=output)
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model
