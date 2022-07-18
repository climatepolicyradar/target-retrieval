"""Create a single file from pdf2text outputs that's easier to read into Python and filter in one go."""

import json
from pathlib import Path
from typing import List
import os

from tqdm.auto import tqdm
from dotenv import find_dotenv, load_dotenv
import click
import pandas as pd

from text_preprocess import TextProcessor


def pdf2text_output_to_jsonl(json_path: Path) -> List[dict]:
    """
    Convert one pdf2text json output into a jsonl list with an object per text block.
    Each text block object has the following fields:
        - md5hash: str
        - page number (starts at 1): int
        - text_block_id: str
        - text: str

    """
    with open(json_path, "r") as f:
        data = json.load(f)

    jsonl_data = []

    md5hash = data["md5hash"]

    for page in data["pages"]:
        page_num = page["page_id"] + 1

        for text_block in page["text_blocks"]:
            jsonl_data.append(
                {
                    "md5hash": md5hash,
                    "page_num": page_num,
                    "text_block_id": text_block["text_block_id"],
                    "text": "".join(text_block["text"]).strip(),
                }
            )

    return jsonl_data


def jsonl_to_dataframe(jsonl_path: Path) -> pd.DataFrame:
    """Load the jsonl file produced when running this script as a CLI to a pandas dataframe."""
    with open(jsonl_path, "r") as f:
        data = json.load(f)

    return pd.DataFrame.from_records(data)


@click.command()
@click.argument("output_path", type=click.Path(dir_okay=False, path_type=str))
def create_jsonl_file_from_pdf2text_outputs(output_path: str) -> None:
    """
    Create a jsonl file from all pdf2text outputs in a directory.
    """
    pdf2text_output_dir = Path(os.environ["DIR_PDF2TEXT_OUTPUTS"])
    pdf2text_json_paths = list(pdf2text_output_dir.glob("*.json"))
    jsonl_data = []

    print("Converting pdf2text output JSONs to one JSONL file")
    for json_path in tqdm(pdf2text_json_paths):
        jsonl_data += pdf2text_output_to_jsonl(json_path)

    print("Processing text using `text_preprocess.TextProcessor`")
    text_processor = TextProcessor()
    processed_data = text_processor.process(jsonl_data, text_key="text")

    with open(output_path, "w") as f:
        json.dump(processed_data, f)


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    create_jsonl_file_from_pdf2text_outputs()
