from datasets import load_dataset

ds = load_dataset("darthraider/fruit-ripeness-detection-dataset")

print(ds)
print(ds["train"][0])