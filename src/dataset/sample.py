import dataclasses
from typing import Callable

import numpy as np


class Embedding:
    """Represents a basic embedding.

    This class holds a NumPy array representing the embedding data.
    """

    def __init__(self, data: np.ndarray):
        """Initialize the Embedding with a NumPy array.

        Args:
            data: The NumPy array containing embedding values.
        """
        self.data = data

    def is_null(self) -> bool:
        """Return whether this embedding is null.

        Returns:
            False, since this is a regular embedding.
        """
        return False


class NullEmbedding(Embedding):
    """Represents a null embedding.

    This class indicates the absence of embedding data.
    """

    def __init__(self):
        """Initialize the NullEmbedding with an empty NumPy array."""
        super().__init__(np.array([]))

    def is_null(self) -> bool:
        """Return whether this embedding is null.

        Returns:
            True, since this is a null embedding.
        """
        return True


@dataclasses.dataclass
class Sample:
    """Represents a single data sample.

    Holds token IDs, metadata, and both normal and noisy embeddings.

    Attributes:
        sample_id: A unique identifier for the sample.
        input_token_ids: The token IDs for the input text.
        metadata: A dictionary containing additional metadata.
        embeddings: An Embedding object containing the sample's embeddings.
        noisy_embeddings: An Embedding object containing the sample's noisy embeddings.
    """

    sample_id: int
    input_token_ids: np.ndarray
    metadata: dict
    embeddings: Embedding
    noisy_embeddings: Embedding = dataclasses.field(default_factory=NullEmbedding)

    def perturb_embeddings(self, noise_mechanism: Callable) -> None:
        """Perturb the embeddings using the provided noise mechanism.

        Applies the specified noise mechanism to the current embeddings and
        updates the noisy_embeddings field with the new, noisy data.

        Args:
            noise_mechanism: An instance of NoiseMechanism used to generate noise.

        Raises:
            ValueError: If the current embeddings are null.
        """
        # if self.embeddings.is_null():
        #     raise ValueError("Cannot perturb null embeddings.")

        noise = noise_mechanism(self.embeddings.data)
        self.noisy_embeddings = self.embeddings.data + noise
