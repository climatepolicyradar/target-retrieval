# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import re

import click
from dotenv import find_dotenv, load_dotenv
import pandas as pd

from translate import translate_bulk
from text_preprocess import TextProcessor


def import_and_check_sheet(
    sheet_name: str, input_filepath: str, value_lists: pd.DataFrame
) -> pd.DataFrame:
    logger.info(f"Checking and converting sheet {sheet_name}")

    if input_filepath.split(".")[-1] != "xlsx":
        raise ValueError("Input file must be an Excel file")

    df = pd.read_excel(input_filepath, sheet_name=sheet_name)

    if (
        not df["Date of Event "]
        .dropna()
        .apply(lambda year: re.match(r"^\d{4}$", str(year)))
        .all()
    ):
        logger.warning("'Date of Event' column contains non-year values")

    for col in value_lists.columns:
        vals_to_check_for = set(value_lists[col].dropna().unique())
        vals_in_data_column = set(df[col].dropna())  # ignores empty values

        if extra_vals := vals_in_data_column - vals_to_check_for:
            logger.warning(
                f"Column '{col}' contains the following values that are not in the list of unique values used for Google Sheets validation: {extra_vals}"
            )

    # Values manually marked to skip which we didn't want to delete, to be able to come back to them later
    if "Skip" in df.columns:
        df = df.loc[df["Skip"] != "TRUE"].drop(columns=["Skip"])

    assert (
        df[
            "Target Description (Paste entire text of target clause in original language)"
        ]
        .isnull()
        .sum()
        == 0
    ), "Target Description column contains missing values."

    df.columns = [re.sub(r"\(.*\)", "", col).strip() for col in df.columns]
    df["Sheet Name"] = sheet_name

    # These columns must have no empty values
    col_errors = {}

    for col in ["Document ID", "Country Name", "Target Description", "Date of Event"]:
        is_null = df[col].isnull()
        if is_null.any():
            # Add 2 to each index to take into account the header row and that Google Sheets starts counting rows at 1
            col_errors.update({col: [i + 2 for i in df[is_null].index.tolist()]})

    if col_errors:
        raise ValueError(
            f"The following columns for sheet '{sheet_name}' have empty values at these rows: {col_errors}"
        )

    return df


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """
    Runs checks on data sheet from 'Labelled Training Data Set' Google Sheet and converts it into a CSV file.

    Also strips text in brackets from column names for easier parsing later on.
    """

    value_lists = pd.read_excel(input_filepath, sheet_name="Value Lists")
    original_dataset = import_and_check_sheet(
        "verified-docs", input_filepath, value_lists
    )
    nonverified_docs_dataset = import_and_check_sheet(
        "non-verified-docs", input_filepath, value_lists
    )
    non_english_docs = import_and_check_sheet(
        "non-english-language", input_filepath, value_lists
    )

    logger.info(f"Translating {len(non_english_docs)} targets to English")
    non_english_docs = non_english_docs.rename(
        columns={"Target Description": "Target Description (original language)"}
    )
    non_english_docs["Target Description"] = translate_bulk(
        non_english_docs["Target Description (original language)"].tolist()
    )

    non_targets = import_and_check_sheet("non-targets", input_filepath, value_lists)
    non_targets["is_target"] = False

    all_targets = pd.concat(
        [original_dataset, nonverified_docs_dataset, non_english_docs],
        axis=0,
        ignore_index=True,
    )
    all_targets["is_target"] = True

    all_data = pd.concat([all_targets, non_targets], axis=0, ignore_index=True)

    print("Processing text using `text_preprocess.TextProcessor`")
    text_processor = TextProcessor()
    data_processed = text_processor.process_dataframe(all_data, "Target Description")

    data_processed.to_csv(output_filepath, index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

    # DEBUG
    # main("./data/raw/Labelled Training Data Set.xlsx", "./data/interim/targets_data.csv")
