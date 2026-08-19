import json
import os

import pandas as pd
from datasets import load_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROC_DIR = os.path.join(BASE_DIR, "proc")
EXT_DIR = os.path.join(BASE_DIR, "ext")

for folder in [RAW_DIR, PROC_DIR, EXT_DIR]:
    os.makedirs(folder, exist_ok=True)


def get_labels_map(dataset, train_df):
    label_feature = dataset["train"].features["label"]

    if hasattr(label_feature, "names"):
        return {int(i): name for i, name in enumerate(label_feature.names)}

    if "label_text" in train_df.columns:
        labels_df = train_df[["label", "label_text"]].drop_duplicates()
        labels_df = labels_df.sort_values("label")

        return {
            int(row.label): str(row.label_text)
            for row in labels_df.itertuples(index=False)
        }

    labels = sorted(train_df["label"].unique())

    return {int(label): str(label) for label in labels}


def download_and_save_data():
    print("Загрузка датасета BANKING77...")
    dataset = load_dataset("mteb/banking77")

    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    print("Сохранение сырых данных в 'raw/'...")
    train_df.to_csv(os.path.join(RAW_DIR, "train_full.csv"), index=False)
    test_df.to_csv(os.path.join(RAW_DIR, "test_full.csv"), index=False)

    print("Сохранение карты интентов в 'ext/'...")
    labels_map = get_labels_map(dataset, train_df)

    with open(os.path.join(EXT_DIR, "intents_map.json"), "w", encoding="utf-8") as f:
        json.dump(labels_map, f, indent=4, ensure_ascii=False)

    print("Создание мини-среза (10 классов) для тестов в 'proc/'...")
    target_labels = list(range(10))

    train_mini = train_df[train_df["label"].isin(target_labels)]
    test_mini = test_df[test_df["label"].isin(target_labels)]

    train_mini.to_csv(os.path.join(PROC_DIR, "train_mini_10.csv"), index=False)
    test_mini.to_csv(os.path.join(PROC_DIR, "test_mini_10.csv"), index=False)

    print("\nВсе данные подготовлены и разложены по полочкам!")


if __name__ == "__main__":
    download_and_save_data()
