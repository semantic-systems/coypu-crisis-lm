""" Create huggingface Dataset class for Crisis tweets """

import csv
import os

import datasets


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

    _URL = {"train": "..data/data/all_data_en/crisis_consolidated_humanitarian_filtered_lang_en_train.csv",
            "val": "..data/data/all_data_en/crisis_consolidated_humanitarian_filtered_lang_en_dev.csv",
            "test": "..data/data/all_data_en/crisis_consolidated_humanitarian_filtered_lang_en_test.csv"}

    train_data_label = ['wildfire', 'earthquake', 'flood', 'typhoon', 'shooting', 'bombing', 'pandemic', 'explosion', 'storm', 'fire', 'hostage', 'tornado']
    BUILDER_CONFIGS = [
        CrisisBenchConfig(
            name="crisis_bench",
            version=datasets.Version("1.0.0"),
            description="CrisisBench Dataset",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=['wildfire', 'earthquake', 'flood', 'typhoon',
                                                                 'shooting', 'bombing', 'pandemic', 'explosion',
                                                                 'storm', 'fire', 'hostage', 'tornado']),
                }
            ),
            supervised_keys=None,
            homepage="http://dcs.gla.ac.uk/~richardm/TREC_IS/2020/data.html",
            citation=_CITATION,
            task_templates=[TextClassification(text_column="text", label_column="label")],
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(self._URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": os.path.join(data_dir["train"])}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": os.path.join(data_dir["test"])}
            )
        ]

    def _generate_examples(self, filepath):
        """Generate Trec IS examples."""
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file, quotechar='"', delimiter=";", quoting=csv.QUOTE_ALL, skipinitialspace=True
            )
            # remove header from the generator
            _ = next(csv_reader)
            for id_, row in enumerate(csv_reader):
                _, num, _, _, _, label, _, text = row
                if num not in ["TRECIS-CTIT-H-086", "TRECIS-CTIT-H-117"]:
                    yield id_, {"text": text, "label": label}