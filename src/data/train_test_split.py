"""
Create a train-test-validation split.
TODO: at the moment this only uses positive examples. This should be updated to use negatives which have been mined from parsed documents.
"""

from pathlib import Path

import pandas as pd
import click


@click.command()
@click.argument("input_path", type=click.Path(dir_okay=False, exists=True))
@click.argument(
    "output_dir", type=click.Path(file_okay=False, exists=True, path_type=Path)
)
def main(input_path, output_dir):
    targets_data = pd.read_csv(input_path)

    def coerce_to_integer_string(val) -> str:
        """Ensure document IDs are all strings with no decimal places in their numbers."""
        try:
            return str(int(float(val)))
        except Exception:
            return str(val)

    targets_data["Document ID"] = targets_data["Document ID"].apply(
        coerce_to_integer_string
    )

    # These document IDs have been selected so in total 43 targets are in the validation and test sets.
    # All of these document IDs are from the 'verified-docs' sheet.
    doc_ids = dict()

    doc_ids["validation"] = {"10264", "8614", "8756", "10195", "10228", "10265"}

    doc_ids["test"] = {"4815 part 1", "9532", "9741", "10372", "10219"}

    doc_ids["train"] = (
        set(
            targets_data.loc[
                targets_data["Sheet Name"] != "non-targets", "Document ID"
            ].tolist()
        )
        - doc_ids["validation"]
        - doc_ids["test"]
    )

    cols_to_keep_and_rename = {
        "Document ID": "Document ID",
        "Country Name": "Country Name",
        "Date of Event": "Document Date",
        "Sheet Name": "Sheet Name",
        "Target Description": "text",
        "Document md5sum": "md5hash",
    }

    for split_name, split_doc_ids in doc_ids.items():
        split_data = targets_data.loc[
            targets_data["Document ID"].isin(split_doc_ids),
            list(cols_to_keep_and_rename.keys()),
        ].rename(columns=cols_to_keep_and_rename)
        split_data.to_csv(output_dir / f"{split_name}_positives.csv", index=False)


if __name__ == "__main__":
    main()
