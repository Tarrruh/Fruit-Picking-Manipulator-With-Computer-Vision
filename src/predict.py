import numpy as np
import tensorflow as tf
from PIL import Image

model_path = "fruit_ripeness_model.keras"
image = 224

model = tf.keras.models.load_model(model_path)

def predict_image(image_path):
    img = Image.open(image_path).resize((image, image))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction >= 0.5:
        print("Ripe")
    else:
        print("Unripe")

# Change the image path here
predict_image("test_fruit.jpg")