from pathlib import Path
from typing import Optional, List
import math
import click
import json

from Levenshtein import ratio
from tqdm.auto import tqdm
import pandas as pd

from pdf2text import jsonl_to_dataframe


class NegativesSampler:
    def __init__(self, pdf2text_jsonl_path: Path, random_state: int):
        self.pdf2text_data = jsonl_to_dataframe(pdf2text_jsonl_path)
        self._rnd = random_state

    def sample_negatives_levenshtein(
        self,
        target_list: List[str],
        n: int,
        levenshtein_ratio_threshold: float = 0.95,
        document_md5_sums: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Randomly sample negatives, and filter out passages with a levenshtein ratio >= to `levenshtein_ratio_threshold`.
        """

        safety_factor = 2

        if document_md5_sums:
            num_sample_per_md5_sum = math.ceil(
                n * safety_factor / len(document_md5_sums)
            )

            candidates = (
                self.pdf2text_data[
                    self.pdf2text_data["md5hash"].isin(document_md5_sums)
                ]
                .groupby("md5hash", group_keys=False)
                .apply(
                    lambda x: x.sample(num_sample_per_md5_sum, random_state=self._rnd)
                )
            )

        else:
            candidates = self.pdf2text_data.sample(
                int(n * safety_factor), random_state=self._rnd
            ).copy(deep=True)

        idxs_drop = set()

        print(
            f"Comparing {len(target_list)} targets to {int(n*safety_factor)} negatives"
        )

        # compare each target passage to all text blocks in a pairwise fashion, and set text blocks that
        # are similar to any of the targets to be dropped from the list of candidates
        for target in tqdm(target_list):
            for idx, row in candidates.iterrows():
                if ratio(target, row["text"]) >= levenshtein_ratio_threshold:
                    idxs_drop = idxs_drop | {idx}

        candidates = candidates.drop(idxs_drop)  # type: ignore

        if len(candidates) >= n:
            return candidates.sample(n, random_state=self._rnd)
        else:
            raise Exception("TODO: raise safety factor and try again")


def create_summary_stats(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> dict:
    """Produce a dictionary with some descriptive statistics of the train, test and validation sets for export to a JSON file."""

    summary_stats = {
        "train": {
            "num_targets": len(train.loc[train["is_target"]]),
            "num_non_targets": len(train.loc[~train["is_target"]]),
            "num_total": len(train),
        },
        "validation": {
            "num_targets": len(validation.loc[validation["is_target"]]),
            "num_non_targets": len(validation.loc[~validation["is_target"]]),
            "num_total": len(validation),
        },
        "test": {
            "num_targets": len(test.loc[test["is_target"]]),
            "num_non_targets": len(test.loc[~test["is_target"]]),
            "num_total": len(test),
        },
    }

    return summary_stats


@click.command()
@click.argument("jsonl_path", type=click.Path(dir_okay=False, path_type=Path))
@click.argument("data_folder", type=click.Path(file_okay=False, path_type=Path))
def main(jsonl_path: Path, data_folder: Path):

    # Import data
    train_positives = pd.read_csv(data_folder / "train_positives.csv")
    validation_positives = pd.read_csv(data_folder / "validation_positives.csv")
    test_positives = pd.read_csv(data_folder / "test_positives.csv")

    for _df in train_positives, validation_positives, test_positives:
        _df["is_target"] = True

    # Parameters
    TRAIN_N_NEGATIVES = int(len(train_positives) * 2)
    TRAIN_LEVENSHTEIN_RATIO_THRESHOLD = 0.95
    VALIDATION_TEST_N_NEGATIVES = {
        "in_documents": int(len(validation_positives) * 2),
        "random": int(len(validation_positives) * 2),
    }

    # Sample negatives for training: ignore which document negatives are from
    print("Sampling negatives for training set")
    train_sampler = NegativesSampler(
        jsonl_path,
        random_state=42,
    )

    train_negatives = train_sampler.sample_negatives_levenshtein(
        target_list=train_positives["text"].unique().tolist(),
        n=TRAIN_N_NEGATIVES,
        levenshtein_ratio_threshold=TRAIN_LEVENSHTEIN_RATIO_THRESHOLD,
    )
    train_negatives["is_target"] = False

    train = pd.concat([train_positives, train_negatives], ignore_index=True)

    # Sample negatives for validation set: include negatives from documents that are in the validation and
    # test sets, and also some negatives outside of this
    print("Sampling negatives for validation set")
    validation_sampler = NegativesSampler(
        jsonl_path,
        random_state=210,
    )

    validation_negatives_in_documents = validation_sampler.sample_negatives_levenshtein(
        target_list=validation_positives["text"].unique().tolist(),
        n=VALIDATION_TEST_N_NEGATIVES["in_documents"],
        document_md5_sums=validation_positives["md5hash"].unique().tolist(),
    )

    validation_negatives_random = validation_sampler.sample_negatives_levenshtein(
        target_list=validation_positives["text"].unique().tolist(),
        n=VALIDATION_TEST_N_NEGATIVES["random"],
    )

    validation_negatives_in_documents["is_target"] = False
    validation_negatives_random["is_target"] = False

    validation = pd.concat(
        [
            validation_positives,
            validation_negatives_in_documents,
            validation_negatives_random,
        ],
        ignore_index=True,
    )

    # Identical sampling for test set
    print("Sampling negatives for test set")
    test_sampler = NegativesSampler(
        jsonl_path,
        random_state=420,
    )

    test_negatives_in_documents = test_sampler.sample_negatives_levenshtein(
        target_list=test_positives["text"].unique().tolist(),
        n=VALIDATION_TEST_N_NEGATIVES["in_documents"],
        document_md5_sums=test_positives["md5hash"].unique().tolist(),
    )

    test_negatives_random = test_sampler.sample_negatives_levenshtein(
        target_list=test_positives["text"].unique().tolist(),
        n=VALIDATION_TEST_N_NEGATIVES["random"],
    )

    test_negatives_in_documents["is_target"] = False
    test_negatives_random["is_target"] = False

    test = pd.concat(
        [test_positives, test_negatives_in_documents, test_negatives_random],
        ignore_index=True,
    )

    summary_stats = create_summary_stats(train, validation, test)
    with open(data_folder / "summary_stats.json", "w") as f:
        json.dump(summary_stats, f, indent=4)

    # Export data
    train.to_csv(data_folder / "train.csv", index=False)
    validation.to_csv(data_folder / "validation.csv", index=False)
    test.to_csv(data_folder / "test.csv", index=False)


if __name__ == "__main__":
    main()
