from pathlib import Path
from abc import ABC
from typing import List

from dotenv import find_dotenv, load_dotenv
import pandas as pd

load_dotenv(find_dotenv())

project_dir = Path(__file__).resolve().parents[2]


class ClassificationDataset(ABC):
    def __init__(self, path: Path):
        self.data = pd.read_csv(path)

    @property
    def _target_col(self) -> str:
        return "is_target"

    @property
    def _text_col(self) -> str:
        return "text"

    @property
    def targets(self) -> List[int]:
        """Classification targets in (0,1)."""
        return (self.data[self._target_col] * 1).tolist()

    @property
    def text(self) -> List[str]:
        return self.data[self._text_col].tolist()


TRAIN = ClassificationDataset(project_dir / "data" / "processed" / "train.csv")
VALIDATION = ClassificationDataset(
    project_dir / "data" / "processed" / "validation.csv"
)
VALIDATION_LARGE = ClassificationDataset(
    project_dir / "data" / "processed" / "validation_large.csv"
)
TEST = ClassificationDataset(project_dir / "data" / "processed" / "test.csv")
TEST_LARGE = ClassificationDataset(
    project_dir / "data" / "processed" / "test_large.csv"
)
