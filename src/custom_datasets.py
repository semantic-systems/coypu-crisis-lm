""" Create huggingface Dataset class for Crisis tweets """

import csv
import pprint
import sys

import datasets

from src.TweetNormalizer import normalizeTweet


_CITATION = """\
"""

_DESCRIPTION = """\
CrisisBench
"""


class CrisisBenchConfig(datasets.BuilderConfig):
    """BuilderConfig for CrisisBench."""

    def __init__(self, **kwargs):
        """BuilderConfig for CrisisBench.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CrisisBenchConfig, self).__init__(**kwargs)


class CrisisBenchDataset(datasets.GeneratorBasedBuilder):
    """CrisisBench Dataset"""
    def __init__(self, class_labels):
        self.BUILDER_CONFIGS = [
        CrisisBenchConfig(
            name="crisis_bench",
            version=datasets.Version("1.0.0"),
            description="CrisisBench Dataset",
        ),
        ]

        self.class_labels = class_labels

    def _info(self):
        return datasets.DatasetInfo(
            description="CrisisBench Corpus",
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=self.class_labels),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/firojalam/crisis_datasets_benchmarks",
            citation="Firoj Alam, Hassan Sajjad, Muhammad Imran and Ferda Ofli, CrisisBench: Benchmarking Crisis-related Social Media Datasets for Humanitarian Information Processing, In ICWSM, 2021.",
            # task_templates=[TextClassification(text_column="text", label_column="label")],
        )

    def _split_generators(self):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": self.data_paths["train"]}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": self.data_paths["dev"]}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": self.data_paths["test"]}
            )
        ]

    def _generate_examples(self, filepath):
        """Generate CrisiBench samples."""

        if filepath.endswith('.tsv'):
            delim = '\t'
        elif filepath.endswith('.csv'):
            delim = ','
        else:
            sys.exit("Unknown file format. Make sure to use a .tsv or .csv dataset file.")

        label_enum_dict = {cls: i for i, cls in enumerate(self.class_labels)}

        with open(filepath, "r", newline=None, encoding='utf-8', errors='replace') as f:
            next(f)  # skip head col
            for id, line in enumerate(f):
                line = line.strip()
                if line == "":
                    continue
                row = line.split(delim)

                text = row[3].strip()
                text = normalizeTweet(text)

                label = row[6]
                label = label_enum_dict[label]

                yield id, {"text": text, "label": label}


