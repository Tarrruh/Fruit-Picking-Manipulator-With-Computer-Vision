from data_loader import load_data
from model import build_model

epochs = 10

train_ds, val_ds = load_data()
model = build_model()

model.summary()

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

model.save("fruit_ripeness_model.keras")