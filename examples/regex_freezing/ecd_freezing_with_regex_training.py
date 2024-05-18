import logging
import os

import pandas as pd
from datasets import load_dataset

from ludwig.api import LudwigModel

dataset = load_dataset("beans")


train_df = pd.DataFrame(
    {"image_path": [f"train_{i}.jpg" for i in range(len(dataset["train"]))], "label": dataset["train"]["labels"]}
)
test_df = pd.DataFrame(
    {"image_path": [f"test_{i}.jpg" for i in range(len(dataset["test"]))], "label": dataset["test"]["labels"]}
)


os.makedirs("train_images", exist_ok=True)
os.makedirs("test_images", exist_ok=True)

for i, img in enumerate(dataset["train"]["image"]):
    img.save(f"train_images/train_{i}.jpg")
for i, img in enumerate(dataset["test"]["image"]):
    img.save(f"test_images/test_{i}.jpg")


train_df["image_path"] = train_df["image_path"].apply(lambda x: os.path.join("train_images", x))
test_df["image_path"] = test_df["image_path"].apply(lambda x: os.path.join("test_images", x))


train_df.to_csv("beans_train.csv", index=False)
test_df.to_csv("beans_test.csv", index=False)


config = {
    "input_features": [
        {
            "name": "image_path",
            "type": "image",
            "encoder": {"type": "resnet", "use_pretrained": True, "trainable": True},
        }
    ],
    "output_features": [{"name": "label", "type": "category"}],
    "trainer": {"epochs": 1, "batch_size": 5, "layers_to_freeze_regex": r"(layer1\.0\.*|layer2\.0\.*)"},
}


model = LudwigModel(config, logging_level=logging.INFO)


train_stats = model.train(dataset="beans_train.csv")


model.save("beans_model")

eval_stats, predictions, output_directory = model.evaluate(dataset="beans_test.csv")


print("Training Statistics: ", train_stats)
print("Evaluation Statistics: ", eval_stats)
