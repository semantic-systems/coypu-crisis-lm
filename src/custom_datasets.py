""" Create huggingface Dataset class for Crisis tweets """

import os
import sys

import datasets

from src.TweetNormalizer import normalizeTweet


_CITATION = """\@inproceedings{alam2020standardizing,
  title={CrisisBench: Benchmarking Crisis-related Social Media Datasets for Humanitarian Information Processing},
  author={Alam, Firoj and Sajjad, Hassan and Imran, Muhammad and Ofli, Ferda},
  booktitle={Proceedings of the International AAAI Conference on Web and Social Media},
  series = {ICWSM~'21},
 month={May}, 
 pages={923-932},  
 number={1},
 volume={15}, 
 url={https://ojs.aaai.org/index.php/ICWSM/article/view/18115},
 year={2021}
}
"""

_DESCRIPTION = """\
The CrisisBench dataset consists data from several different data sources such as CrisisLex (CrisisLex26, CrisisLex6), CrisisNLP, SWDM2013, ISCRAM13, Disaster Response Data (DRD), Disasters on Social Media (DSM), CrisisMMD and data from AIDR. 
"""

_LICENSE = """CC BY-NC-SA 4.0"""

_HOMEPAGE = """https://github.com/firojalam/crisis_datasets_benchmarks"""

_URL = """https://crisisnlp.qcri.org/data/crisis_datasets_benchmarks/crisis_datasets_benchmarks_v1.0.tar.gz"""
_DATA_SUBFOLDER = """all_data_en"""


class CrisisBenchBuilderConfig(datasets.BuilderConfig):
    def __init__(self, name, description, classes):
        datasets.BuilderConfig.__init__(self, name, description)
        self.classes = classes


class CrisisBenchDataset(datasets.GeneratorBasedBuilder):
    """CrisisBench Dataset"""
    BUILDER_CONFIGS = [
        CrisisBenchBuilderConfig(
            name="humanitarian",
            description="Classification task for humanitarian crisis type.",
            classes=['affected_individual', 'caution_and_advice', 'displaced_and_evacuations',
                     'donation_and_volunteering', 'infrastructure_and_utilities_damage', 'injured_or_dead_people', 'missing_and_found_people', 'not_humanitarian', 'requests_or_needs', 'response_efforts', 'sympathy_and_support'],
        ),
        CrisisBenchBuilderConfig(
            name="informativeness",
            description="Detection task for crisis tweets.",
            classes=['informative', 'not_informative'],
        ),
        ]

    DEFAULT_CONFIG_NAME = "informativeness"

    def _info(self):
        if self.config.name == "humanitarian":
            features = datasets.Features(
                    {
                        "text": datasets.Value("string"),
                        "label": datasets.features.ClassLabel(names=self.config.classes),
                    }
                )
        else:
            features = datasets.Features(
                    {
                        "text": datasets.Value("string"),
                        "label": datasets.features.ClassLabel(names=self.config.classes),
                    }
                )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            # task_templates=[TextClassification(text_column="text", label_column="label")],
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": os.path.join(data_dir, _DATA_SUBFOLDER,
                                                                                f"crisis_consolidated_{self.config.name}_filtered_lang_en_train.tsv")}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": os.path.join(data_dir, _DATA_SUBFOLDER,
                                                                                f"crisis_consolidated_{self.config.name}_filtered_lang_en_dev.tsv")}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": os.path.join(data_dir, _DATA_SUBFOLDER,
                                                                                f"crisis_consolidated_{self.config.name}_filtered_lang_en_test.tsv")}
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

        label_enum_dict = {cls: i for i, cls in enumerate(self.config.classes)}

        with open(filepath, "r", newline=None, encoding='utf-8', errors='replace') as f:
            next(f)  # skip head col
            for i, line in enumerate(f):
                line = line.strip()
                if line == "":
                    continue
                row = line.split(delim)

                text = row[3].strip()
                text = normalizeTweet(text)

                label = row[6]
                label = label_enum_dict[label]

                yield i, {"text": text, "label": label}


