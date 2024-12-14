from src.data_preparation import prepare_dataset
from src.model import build_model
from src.config import EPOCHS, BATCH_SIZE

def train_model(start_date, end_date):
    X_images, X_weather, Y = prepare_dataset(start_date, end_date)
    split_idx = int(0.8 * len(Y))
    X_train_img, X_train_w, Y_train = X_images[:split_idx], X_weather[:split_idx], Y[:split_idx]
    X_val_img, X_val_w, Y_val = X_images[split_idx:], X_weather[split_idx:], Y[split_idx:]

    model = build_model()
    model.fit([X_train_img, X_train_w], Y_train,
              validation_data=([X_val_img, X_val_w], Y_val),
              epochs=EPOCHS, batch_size=BATCH_SIZE)
    return model
