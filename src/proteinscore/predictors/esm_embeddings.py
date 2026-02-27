"""
ESM-2 Embeddings Module (HuggingFace Implementation)

Provides protein embeddings using ESM-2 models WITHOUT requiring fair-esm library.
Uses HuggingFace Transformers which has native ESM-2 support.

Supported models (all CPU-compatible):
- esm2_t6_8M_UR50D: 8M params, fastest, good for screening
- esm2_t12_35M_UR50D: 35M params, better accuracy, still fast on CPU
- esm2_t30_150M_UR50D: 150M params, high accuracy (may need more RAM)

Reference: https://huggingface.co/facebook/esm2_t6_8M_UR50D
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Model configurations (HuggingFace model IDs)
ESM2_MODELS = {
    "8M": "facebook/esm2_t6_8M_UR50D",      # 6 layers, 320 dim
    "35M": "facebook/esm2_t12_35M_UR50D",   # 12 layers, 480 dim
    "150M": "facebook/esm2_t30_150M_UR50D", # 30 layers, 640 dim
}

# Embedding dimensions per model
ESM2_DIMS = {
    "8M": 320,
    "35M": 480,
    "150M": 640,
}

# Default model for CPU usage (best speed/accuracy tradeoff)
DEFAULT_MODEL = "35M"


class ESM2Embedder:
    """
    ESM-2 protein embeddings using HuggingFace Transformers.

    This implementation does NOT require the fair-esm library.
    Uses transformers library with automatic model download from HuggingFace Hub.

    Example:
        embedder = ESM2Embedder(model_size="35M")
        embedding = embedder.get_embedding("MKTVRQERLKSIVRILERSKEPVSGAQ")
    """

    _instance: "ESM2Embedder | None" = None
    _model = None
    _tokenizer = None
    _model_size: str | None = None

    def __new__(cls, model_size: str = DEFAULT_MODEL) -> "ESM2Embedder":
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None or cls._model_size != model_size:
            cls._instance = super().__new__(cls)
            cls._model_size = model_size
        return cls._instance

    def __init__(self, model_size: str = DEFAULT_MODEL) -> None:
        """
        Initialize ESM-2 embedder.

        Args:
            model_size: One of "8M", "35M", "150M"
        """
        self.model_size = model_size
        self.model_id = ESM2_MODELS.get(model_size, ESM2_MODELS[DEFAULT_MODEL])
        self.embedding_dim = ESM2_DIMS.get(model_size, ESM2_DIMS[DEFAULT_MODEL])
        self._loaded = False

    def _ensure_loaded(self) -> bool:
        """
        Lazy-load the model and tokenizer.

        Returns True if model is available, False otherwise.
        """
        if self._loaded and self._model is not None:
            return True

        try:
            from transformers import AutoModel, AutoTokenizer
            import torch

            logger.info(f"Loading ESM-2 model: {self.model_id}")

            # Load tokenizer
            ESM2Embedder._tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            # Load model with CPU settings
            ESM2Embedder._model = AutoModel.from_pretrained(
                self.model_id,
            )
            ESM2Embedder._model.eval()  # Set to evaluation mode

            self._loaded = True
            logger.info(f"ESM-2 model loaded successfully (dim={self.embedding_dim})")
            return True

        except ImportError as e:
            logger.warning(
                f"transformers or torch not installed. "
                f"Install with: pip install transformers torch. Error: {e}"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to load ESM-2 model: {e}")
            return False

    def is_available(self) -> bool:
        """Check if ESM-2 embeddings are available."""
        return self._ensure_loaded()

    def get_embedding(
        self,
        sequence: str,
        pooling: str = "mean",
    ) -> "NDArray[np.float32] | None":
        """
        Get ESM-2 embedding for a protein sequence.

        Args:
            sequence: Protein sequence (amino acid letters)
            pooling: Pooling strategy - "mean" (default), "cls", or "last"

        Returns:
            Numpy array of shape (embedding_dim,) or None if unavailable
        """
        if not self._ensure_loaded():
            return None

        try:
            import torch

            # Clean sequence (remove spaces, uppercase)
            sequence = sequence.upper().replace(" ", "")

            # Truncate if too long (ESM-2 max is 1024 tokens)
            max_len = 1022  # Leave room for special tokens
            if len(sequence) > max_len:
                logger.warning(f"Sequence truncated from {len(sequence)} to {max_len}")
                sequence = sequence[:max_len]

            # Tokenize
            inputs = self._tokenizer(
                sequence,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=1024,
            )

            # Get embeddings
            with torch.no_grad():
                outputs = self._model(**inputs)
                hidden_states = outputs.last_hidden_state  # (1, seq_len, dim)

            # Apply pooling
            if pooling == "cls":
                # Use CLS token (first position)
                embedding = hidden_states[0, 0, :].numpy()
            elif pooling == "last":
                # Use last token before EOS
                embedding = hidden_states[0, -2, :].numpy()
            else:  # mean pooling (default, most robust)
                # Average over all positions (excluding special tokens)
                embedding = hidden_states[0, 1:-1, :].mean(dim=0).numpy()

            return embedding.astype(np.float32)

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    def get_per_residue_embeddings(
        self,
        sequence: str,
    ) -> "NDArray[np.float32] | None":
        """
        Get per-residue embeddings for a protein sequence.

        Returns embeddings for each amino acid position, useful for
        position-specific predictions.

        Args:
            sequence: Protein sequence

        Returns:
            Numpy array of shape (seq_len, embedding_dim) or None
        """
        if not self._ensure_loaded():
            return None

        try:
            import torch

            sequence = sequence.upper().replace(" ", "")
            max_len = 1022
            if len(sequence) > max_len:
                sequence = sequence[:max_len]

            inputs = self._tokenizer(
                sequence,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=1024,
            )

            with torch.no_grad():
                outputs = self._model(**inputs)
                hidden_states = outputs.last_hidden_state

            # Remove special tokens (first and last)
            per_residue = hidden_states[0, 1:-1, :].numpy()

            return per_residue.astype(np.float32)

        except Exception as e:
            logger.error(f"Failed to generate per-residue embeddings: {e}")
            return None

    def get_attention_contacts(
        self,
        sequence: str,
        layer: int = -1,
    ) -> "NDArray[np.float32] | None":
        """
        Get attention-based contact predictions.

        ESM-2 attention maps correlate with protein contacts.

        Args:
            sequence: Protein sequence
            layer: Which layer's attention to use (-1 for last)

        Returns:
            Contact map of shape (seq_len, seq_len) or None
        """
        if not self._ensure_loaded():
            return None

        try:
            import torch

            sequence = sequence.upper().replace(" ", "")
            max_len = 1022
            if len(sequence) > max_len:
                sequence = sequence[:max_len]

            inputs = self._tokenizer(
                sequence,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=1024,
            )

            with torch.no_grad():
                outputs = self._model(**inputs, output_attentions=True)
                attentions = outputs.attentions  # Tuple of attention tensors

            # Get specified layer
            attn = attentions[layer]  # (1, heads, seq, seq)

            # Average over heads and symmetrize
            attn_map = attn[0].mean(dim=0).numpy()  # (seq, seq)
            attn_map = (attn_map + attn_map.T) / 2

            # Remove special tokens
            attn_map = attn_map[1:-1, 1:-1]

            return attn_map.astype(np.float32)

        except Exception as e:
            logger.error(f"Failed to generate attention contacts: {e}")
            return None


# Convenience functions
@lru_cache(maxsize=1)
def get_embedder(model_size: str = DEFAULT_MODEL) -> ESM2Embedder:
    """Get cached ESM-2 embedder instance."""
    return ESM2Embedder(model_size=model_size)


def get_esm2_embedding(
    sequence: str,
    model_size: str = DEFAULT_MODEL,
    pooling: str = "mean",
) -> "NDArray[np.float32] | None":
    """
    Convenience function to get ESM-2 embedding.

    Args:
        sequence: Protein sequence
        model_size: "8M", "35M", or "150M"
        pooling: "mean", "cls", or "last"

    Returns:
        Embedding array or None if unavailable
    """
    embedder = get_embedder(model_size)
    return embedder.get_embedding(sequence, pooling=pooling)


def is_esm2_available(model_size: str = DEFAULT_MODEL) -> bool:
    """Check if ESM-2 is available (transformers + torch installed)."""
    embedder = get_embedder(model_size)
    return embedder.is_available()


# Feature extraction for stability prediction
def extract_stability_features(
    sequence: str,
    model_size: str = DEFAULT_MODEL,
) -> dict[str, float] | None:
    """
    Extract stability-relevant features from ESM-2 embeddings.

    Features derived from embedding statistics that correlate with Tm:
    - Mean embedding values
    - Embedding variance (structural diversity)
    - Specific dimension statistics

    Args:
        sequence: Protein sequence
        model_size: ESM-2 model size

    Returns:
        Dictionary of stability features or None
    """
    embedder = get_embedder(model_size)

    embedding = embedder.get_embedding(sequence, pooling="mean")
    if embedding is None:
        return None

    per_residue = embedder.get_per_residue_embeddings(sequence)
    if per_residue is None:
        return None

    # Extract features
    features = {
        # Global embedding statistics
        "emb_mean": float(np.mean(embedding)),
        "emb_std": float(np.std(embedding)),
        "emb_max": float(np.max(embedding)),
        "emb_min": float(np.min(embedding)),
        "emb_range": float(np.max(embedding) - np.min(embedding)),

        # Per-residue variability (structural diversity)
        "residue_mean_std": float(np.mean(np.std(per_residue, axis=0))),
        "residue_max_std": float(np.max(np.std(per_residue, axis=0))),

        # Embedding norm (overall "magnitude")
        "emb_norm": float(np.linalg.norm(embedding)),

        # Quartile statistics
        "emb_q25": float(np.percentile(embedding, 25)),
        "emb_q75": float(np.percentile(embedding, 75)),
        "emb_iqr": float(np.percentile(embedding, 75) - np.percentile(embedding, 25)),

        # Skewness and kurtosis (distribution shape)
        "emb_skew": float(_safe_skew(embedding)),
        "emb_kurtosis": float(_safe_kurtosis(embedding)),

        # Position-based features
        "n_term_mean": float(np.mean(per_residue[:10])) if len(per_residue) > 10 else float(np.mean(per_residue)),
        "c_term_mean": float(np.mean(per_residue[-10:])) if len(per_residue) > 10 else float(np.mean(per_residue)),
        "core_mean": float(np.mean(per_residue[len(per_residue)//4:3*len(per_residue)//4])),
    }

    return features


def _safe_skew(arr: "NDArray") -> float:
    """Calculate skewness safely."""
    try:
        from scipy.stats import skew
        return float(skew(arr))
    except ImportError:
        # Fallback: simple skewness approximation
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-10:
            return 0.0
        return float(np.mean(((arr - mean) / std) ** 3))


def _safe_kurtosis(arr: "NDArray") -> float:
    """Calculate kurtosis safely."""
    try:
        from scipy.stats import kurtosis
        return float(kurtosis(arr))
    except ImportError:
        # Fallback: simple kurtosis approximation
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-10:
            return 0.0
        return float(np.mean(((arr - mean) / std) ** 4) - 3)
