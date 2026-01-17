import numpy as np
import tensorflow as tf
from datasets import load_dataset

IMG_SIZE = 224
BATCH_SIZE = 32

def load_data():
    ds = load_dataset("darthraider/fruit-ripeness-detection-dataset")

    def preprocess(example):
        image = example["image"].resize((IMG_SIZE, IMG_SIZE))
        image = np.array(image) / 255.0
        label = example["label"]
        return image, label

    def make_tf_dataset(split):
        images, labels = [], []
        for example in ds[split]:
            img, lbl = preprocess(example)
            images.append(img)
            labels.append(lbl)

        return tf.data.Dataset.from_tensor_slices(
            (np.array(images), np.array(labels))
        ).shuffle(500).batch(BATCH_SIZE)

    train_ds = make_tf_dataset("train")
    val_ds = make_tf_dataset("validation")
