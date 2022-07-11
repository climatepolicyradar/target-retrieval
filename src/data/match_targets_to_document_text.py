"""Match targets in dataset to text blocks in parsed PDFs. Not all targets will be matched."""
import os

from tqdm.auto import tqdm
from Levenshtein import ratio
import pandas as pd
import click
from dotenv import find_dotenv, load_dotenv

from opensearch import OpenSearchIndex, _convert_to_bool


def build_match_query(query_text: str, fuzziness: int = 3) -> dict:
    """Build a match query to return candidate matches in OpenSearch for targets

    Args:
        query_text (str)
        fuzziness (int, optional): see opensearch docs. Defaults to 3.

    Returns:
        dict: query
    """

    return {
        "_source": {
            "excludes": [
                "text_embedding",
                "text_block_coords",
                "document_description",
                "document_description_embedding",
            ]
        },
        "query": {
            "match": {
                "text": {
                    "query": query_text,
                    "fuzziness": fuzziness,
                }
            }
        },
    }


def find_targets_in_documents(
    targets_data: pd.DataFrame, opns_conn: OpenSearchIndex
) -> pd.DataFrame:
    """Find targets in documents. For a target to match a document, one of the following criteria must be true:
    * String similarity between target and text block is greater than or equal to 0.95,
    * Target text is a substring of the text block (case insensitive), or
    * Text block is a substring of the target text, and more than 60% of the length of the target text.

    Args:
        targets_data (pd.DataFrame): created using `src/data/make_targets_data.py`
        opns_index (OpenSearchIndex)

    Returns:
        pd.DataFrame: columns "Target text", "Document text", "Text block ID", "Document ID", "String Similarity","Target in text block","Text block in target"
    """
    matched_text = []

    for target_text in tqdm(targets_data["Target Description"].unique().tolist()):
        response_hits = opns_conn.opns.search(
            index="navigator", body=build_match_query(target_text)
        )["hits"]["hits"]

        for hit in response_hits:
            text = hit["_source"]["text"]
            text_block_id = hit["_source"]["text_block_id"]
            document_id = hit["_source"]["document_id"]

            similarity = ratio(target_text, text)
            target_in_text_block = target_text.lower() in text.lower()
            text_block_length_divided_by_target_text_length = len(text) / len(
                target_text
            )
            text_block_in_target = (text.lower() in target_text.lower()) and (
                text_block_length_divided_by_target_text_length > 0.6
            )

            if (similarity > 0.98) or text_block_in_target or target_in_text_block:
                matched_text.append(
                    (
                        target_text,
                        text,
                        text_block_id,
                        document_id,
                        similarity,
                        target_in_text_block,
                        text_block_in_target,
                    )
                )

    return pd.DataFrame(
        matched_text,
        columns=[
            "Target text",
            "Document text",
            "Text block ID",
            "Document ID",
            "String Similarity",
            "Target in text block",
            "Text block in target",
        ],
    )


@click.command()
@click.argument(
    "input-data-path", type=click.Path(exists=True, dir_okay=False), required=True
)
@click.argument("output-data-path", type=click.Path(dir_okay=False), required=True)
def main(input_data_path, output_data_path):
    """Load targets data, find matches in documents, and save results to file."""
    print("Loading targets data")
    targets_data = pd.read_csv(input_data_path)

    print("Finding targets in documents")
    opns_connector = OpenSearchIndex(
        index_name=os.environ["OPENSEARCH_INDEX"],
        url=os.environ["OPENSEARCH_URL"],
        username=os.environ["OPENSEARCH_USER"],
        password=os.environ["OPENSEARCH_PASSWORD"],
        opensearch_connector_kwargs={
            "use_ssl": _convert_to_bool(os.environ["OPENSEARCH_USE_SSL"]),
            "verify_certs": _convert_to_bool(os.environ["OPENSEARCH_VERIFY_CERTS"]),
            "ssl_show_warn": _convert_to_bool(os.environ["OPENSEARCH_SSL_WARNINGS"]),
        },
    )

    targets_matched = find_targets_in_documents(targets_data, opns_connector)
    print(
        f"One or more matching text blocks found for {len(targets_matched['Target text'].unique())} targets."
    )

    targets_matched.to_csv(output_data_path, index=False)


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    main()
