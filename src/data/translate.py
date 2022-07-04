from typing import List
from googletrans import Translator


def translate_bulk(text_list: List[str]) -> List[str]:
    """Translate a list of strings to English."""

    translator = Translator()
    translations = [translator.translate(text, dest="en") for text in text_list]

    return [translation.text for translation in translations]
