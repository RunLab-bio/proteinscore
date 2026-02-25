"""
Base Predictor Interface

Abstract base class for all developability predictors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from proteinscore.config import PredictorConfig

ResultT = TypeVar("ResultT", bound=BaseModel)


class BasePredictor(ABC, Generic[ResultT]):
    """
    Abstract base class for developability predictors.

    All predictors must implement the predict() method and return
    a Pydantic model with a score field in the 0-100 range.
    """

    def __init__(self, config: PredictorConfig | None = None) -> None:
        """
        Initialize the predictor.

        Args:
            config: Predictor-specific configuration
        """
        self.config = config
        self._initialized = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the predictor."""
        ...

    @property
    @abstractmethod
    def method(self) -> str:
        """Method identifier for result attribution."""
        ...

    @property
    def is_initialized(self) -> bool:
        """Check if the predictor is ready for predictions."""
        return self._initialized

    def initialize(self) -> None:
        """
        Perform any necessary initialization.

        Override this method for predictors that need setup
        (e.g., loading models, checking dependencies).
        """
        self._initialized = True

    @abstractmethod
    def predict(
        self,
        sequence: str,
        structure: str | None = None,
    ) -> ResultT:
        """
        Make a prediction for the given sequence.

        Args:
            sequence: Protein sequence (validated, uppercase)
            structure: Optional PDB structure string

        Returns:
            Predictor-specific result model with score in 0-100 range
        """
        ...

    def predict_batch(
        self,
        sequences: list[str],
        structures: list[str | None] | None = None,
    ) -> list[ResultT]:
        """
        Make predictions for multiple sequences.

        Default implementation calls predict() in sequence.
        Override for more efficient batch processing.

        Args:
            sequences: List of protein sequences
            structures: Optional list of PDB structures

        Returns:
            List of predictor-specific result models
        """
        if structures is None:
            structures = [None] * len(sequences)

        return [
            self.predict(seq, struct)
            for seq, struct in zip(sequences, structures)
        ]

    def validate_inputs(
        self,
        sequence: str,
        structure: str | None = None,
    ) -> None:
        """
        Validate inputs before prediction.

        Override to add predictor-specific validation.

        Raises:
            ValidationError: If inputs are invalid
        """
        if not sequence:
            raise ValueError("Sequence cannot be empty")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(method={self.method!r})"
