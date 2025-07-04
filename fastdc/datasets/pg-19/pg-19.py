"""PG-19 language modeling benchmark - a set of books extracted from the Project Gutenberg books library"""


import csv
import os

import datasets


# TODO(pg19): BibTeX citation
_CITATION = """\
@article{raecompressive2019,
  author = {Rae, Jack W and Potapenko, Anna and Jayakumar, Siddhant M and
            Hillier, Chloe and Lillicrap, Timothy P},
  title = {Compressive Transformers for Long-Range Sequence Modelling},
  journal = {arXiv preprint},
  url = {https://arxiv.org/abs/1911.05507},
  year = {2019},
}
"""

# TODO(pg19):
_DESCRIPTION = """\
This repository contains the PG-19 language modeling benchmark.
It includes a set of books extracted from the Project Gutenberg books library, that were published before 1919.
It also contains metadata of book titles and publication dates.
PG-19 is over double the size of the Billion Word benchmark and contains documents that are 20X longer, on average, than the WikiText long-range language modelling benchmark.
Books are partitioned into a train, validation, and test set. Book metadata is stored in metadata.csv which contains (book_id, short_book_title, publication_date).
Unlike prior benchmarks, we do not constrain the vocabulary size --- i.e. mapping rare words to an UNK token --- but instead release the data as an open-vocabulary benchmark. The only processing of the text that has been applied is the removal of boilerplate license text, and the mapping of offensive discriminatory words as specified by Ofcom to placeholder tokens. Users are free to model the data at the character-level, subword-level, or via any mechanism that can model an arbitrary string of text.
To compare models we propose to continue measuring the word-level perplexity, by calculating the total likelihood of the dataset (via any chosen subword vocabulary or character-based scheme) divided by the number of tokens --- specified below in the dataset statistics table.
One could use this dataset for benchmarking long-range language models, or use it to pre-train for other natural language processing tasks which require long-range reasoning, such as LAMBADA or NarrativeQA. We would not recommend using this dataset to train a general-purpose language model, e.g. for applications to a production-system dialogue agent, due to the dated linguistic style of old texts and the inherent biases present in historical writing.
"""


_SPLIT_FILES_PATH = "data/{split}_files.txt"
_ASSET_ROOT_URL = "https://storage.googleapis.com/deepmind-gutenberg/"
_METADATA_URL = _ASSET_ROOT_URL + "metadata.csv"


def flat_map(fn, arr):
    return [el for sub_arr in map(fn, arr) for el in sub_arr]


class Pg19(datasets.GeneratorBasedBuilder):
    """PG-19 dataset - books as plain text extracted from the Project Gutenberg library"""

    # TODO(pg19): Set up version.
    VERSION = datasets.Version("0.1.0")

    def _info(self):
        # TODO(pg19): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "short_book_title": datasets.Value("string"),
                    "publication_date": datasets.Value("int32"),
                    "url": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://github.com/deepmind/pg19",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        splits = ["train", "validation", "test"]
        files = dl_manager.download({split: _SPLIT_FILES_PATH.format(split=split) for split in splits})

        for split, names_file in list(files.items()):
            with open(names_file, encoding="utf-8") as f:
                split_files = f.read().splitlines()
            split_files = sorted(split_files)
            split_files = {
                os.path.splitext(os.path.basename(file))[0]: _ASSET_ROOT_URL + file
                for file in split_files
            }
            files[split] = split_files

        metadata = dl_manager.download(_METADATA_URL)
        downloaded_files = dl_manager.download(files)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "ids": list(downloaded_files["train"]),
                    "metadata": metadata,
                    "files": downloaded_files["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "ids": list(downloaded_files["validation"]),
                    "metadata": metadata,
                    "files": downloaded_files["validation"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "ids": list(downloaded_files["test"]),
                    "metadata": metadata,
                    "files": downloaded_files["test"],
                },
            ),
        ]

    def _generate_examples(self, ids, metadata, files):
        """Yields examples."""
        # TODO(pg19): Yields (key, example) tuples from the dataset

        with open(metadata, encoding="utf-8") as f:
            reader = csv.DictReader(f, fieldnames=["_id", "short_book_title", "publication_date", "url"])
            id2metadata = {row["_id"]: row for row in reader}

        for _id in ids:
            data = id2metadata[_id]
            file = files[_id]

            with open(file, encoding="utf-8") as f:
                text = f.read()

            _id = data["_id"]
            short_book_title = data["short_book_title"]
            publication_date = int(data["publication_date"])
            url = data["url"]

            yield _id, {
                "short_book_title": short_book_title,
                "publication_date": publication_date,
                "url": url,
                "text": text,
            }
