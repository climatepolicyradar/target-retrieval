from typing import List

import pandas as pd


class TextProcessor:
    """
    Pre-processing steps (i.e. before ML) for text.
    """

    @staticmethod
    def _contains_min_words(text: str, min_words: int) -> bool:
        """Return True if the text contains at least `min_words` words."""
        # TODO: better tokeniser
        return len(text.split()) >= min_words

    @staticmethod
    def _all_non_alphabet_characters(text: str) -> bool:
        """Return True if the text contains only non-letter characters."""
        return all(not c.isalpha() for c in text)

    def process(self, data: List[dict], text_key: str) -> List[dict]:
        """
        Process a list of dicts with the text in the key `text_key`.
        """

        MIN_N_WORDS = 3

        data_out = []

        for d in data:
            if self._contains_min_words(
                d[text_key], MIN_N_WORDS
            ) and not self._all_non_alphabet_characters(d[text_key]):

                data_out.append(d)

        return data_out

    def process_dataframe(self, data: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """Process a dataframe with the text in the column `text_col`."""

        data_dict = data.to_dict("records")
        data_processed = self.process(data_dict, text_col)

        return pd.DataFrame.from_records(data_processed)
