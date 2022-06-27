# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv
import pandas as pd


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """
    Runs checks on data sheet from 'Labelled Training Data Set' Google Sheet and converts it into a CSV file.
    """

    logger = logging.getLogger(__name__)
    logger.info("Checking and converting data")

    if input_filepath.split(".")[-1] != "xlsx":
        raise ValueError("Input file must be an Excel file")

    df = pd.read_excel(input_filepath, sheet_name="Training Data Set")
    value_lists = pd.read_excel(input_filepath, sheet_name="Value Lists")
    for col in value_lists.columns:
        vals_to_check_for = set(value_lists[col].dropna().unique())
        vals_in_data_column = set(df[col].dropna())  # ignores empty values

        if extra_vals := vals_in_data_column - vals_to_check_for:
            logger.warning(
                f"Column '{col}' contains the following values that are not in the list of unique values used for Google Sheets validation: {extra_vals}"
            )

    assert (
        df[
            "Target Description (Paste entire text of target clause in original language)"
        ]
        .isnull()
        .sum()
        == 0
    ), "Target Description column contains missing values."

    df.to_csv(output_filepath, index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
