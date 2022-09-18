import csv
import json
import os

import datasets
import pandas as pd

_CITATION = """\
Unknown
"""

_DESCRIPTION = """\
Ini adalah dataloader untuk mengerjakan tugas NLP ya ges ya
"""

_HOMEPAGE = "https://github.com/rayendito/TextClassification"

_LICENSE = "Unknown"

_URLS = {
    "train" : "https://raw.githubusercontent.com/rayendito/TextClassification/main/data_worthcheck/train.csv",
    "test" : "https://raw.githubusercontent.com/rayendito/TextClassification/main/data_worthcheck/test.csv",
    "val" : "https://raw.githubusercontent.com/rayendito/TextClassification/main/data_worthcheck/dev.csv",
}


class SentimentDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.1.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="sentiment_dataset",
            version=VERSION,
            description="Ini adalah dataloader untuk mengerjakan tugas NLP ya ges ya"
        ),
    ]

    DEFAULT_CONFIG_NAME = "sentiment_dataset"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        if self.config.name == "sentiment_dataset":  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.Value("int64")
                }
            )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": _URLS["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": _URLS["val"],
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": _URLS["test"],
                    "split": "test"
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        df = pd.read_csv(filepath).reset_index()
        self._encode_labels(df)
        for row in df.itertuples():
            entry = {
                "text" : row.text_a,
                "label" : row.label_encoded
            }
            yield row.index, entry
    
    def _encode_labels(self, df):
        df['label_encoded'] = [0 if (entry == 'no') else 1 for entry in df['label']]


if __name__ == "__main__":
    datasets.load_dataset(__file__)