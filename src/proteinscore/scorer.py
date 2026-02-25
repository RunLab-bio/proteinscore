"""
ProteinScore Main Scorer

Unified protein developability scorer combining stability, solubility,
aggregation, and immunogenicity into a single 0-100 score.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import TYPE_CHECKING

from proteinscore.config import Config, ScoringWeights, get_default_config
from proteinscore.models import (
    AggregationResult,
    ImmunogenicityResult,
    ProteinScoreResult,
    Recommendation,
    RiskLevel,
    SolubilityResult,
    StabilityResult,
)
from proteinscore.predictors import (
    AggregationPredictor,
    ImmunogenicityPredictor,
    SolubilityPredictor,
    StabilityPredictor,
)
from proteinscore.validation import validate_sequence

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class ProteinScore:
    """
    Unified Protein Developability Scorer.

    Combines four key developability metrics into a single 0-100 score:
    - Stability: Thermodynamic stability prediction
    - Solubility: Expression solubility prediction
    - Aggregation: Aggregation propensity analysis
    - Immunogenicity: T-cell epitope prediction (via RIP API)

    Example:
        >>> scorer = ProteinScore(api_key="rl_...")
        >>> result = scorer.score("MKTAYIAKQRQISFVK...")
        >>> print(f"ProteinScore: {result.total_score}/100")
        >>> print(f"Risk: {result.risk_level}")
    """

    def __init__(
        self,
        api_key: str | None = None,
        weights: ScoringWeights | None = None,
        hla_alleles: list[str] | None = None,
        hla_population: str = "global",
        local_only: bool = False,
        cache_results: bool = True,
        config: Config | None = None,
    ) -> None:
        """
        Initialize ProteinScore scorer.

        Args:
            api_key: RunLab API key for immunogenicity predictions.
                    Uses anonymous tier if not provided.
            weights: Custom scoring weights. Default: equal weights (0.25 each).
            hla_alleles: Specific HLA alleles for immunogenicity.
                        Overrides population selection.
            hla_population: Population for HLA allele selection.
                           Options: global, european, african, asian, hispanic.
            local_only: Skip API calls, use local estimation for immunogenicity.
            cache_results: Cache API responses locally.
            config: Full configuration object (overrides other parameters).
        """
        # Use provided config or create from parameters
        if config is None:
            config = get_default_config()

        self._config = config
        self._weights = weights or config.default_weights
        self._local_only = local_only or config.local_only
        self._cache_enabled = cache_results and config.cache_enabled

        # Resolve API key
        self._api_key = api_key or config.api_key

        # Initialize predictors
        self._stability = StabilityPredictor()
        self._solubility = SolubilityPredictor()
        self._aggregation = AggregationPredictor()
        self._immunogenicity = ImmunogenicityPredictor(
            api_key=None if self._local_only else self._api_key,
            api_base_url=config.api_base_url,
            hla_alleles=hla_alleles or config.hla_alleles,
            hla_population=hla_population or config.hla_population,
        )

        # Cache for results
        self._cache: dict[str, ProteinScoreResult] = {}

        logger.info(
            "ProteinScore initialized (api_key=%s, local_only=%s)",
            "set" if self._api_key else "anonymous",
            self._local_only,
        )

    def score(
        self,
        sequence: str,
        structure: str | None = None,
        name: str | None = None,
    ) -> ProteinScoreResult:
        """
        Calculate developability score for a protein.

        Args:
            sequence: Amino acid sequence (one-letter code).
            structure: Optional PDB structure (path or string).
                      Enables more accurate stability/aggregation predictions.
            name: Optional name for the protein.

        Returns:
            ProteinScoreResult with component scores and overall score.

        Raises:
            InvalidSequenceError: If sequence is invalid.
        """
        # Validate and normalize sequence
        sequence = validate_sequence(sequence)

        # Check cache
        cache_key = self._get_cache_key(sequence, structure)
        if self._cache_enabled and cache_key in self._cache:
            logger.debug("Cache hit for %s", cache_key[:8])
            return self._cache[cache_key]

        # Run all predictors
        stability_result = self._stability.predict(sequence, structure)
        solubility_result = self._solubility.predict(sequence, structure)
        aggregation_result = self._aggregation.predict(sequence, structure)
        immunogenicity_result = self._immunogenicity.predict(sequence, structure)

        # Calculate weighted total score
        total_score = (
            self._weights.stability * stability_result.score +
            self._weights.solubility * solubility_result.score +
            self._weights.aggregation * aggregation_result.score +
            self._weights.immunogenicity * immunogenicity_result.score
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            stability_result,
            solubility_result,
            aggregation_result,
            immunogenicity_result,
        )

        # Build result
        result = ProteinScoreResult(
            total_score=round(total_score, 1),
            stability=stability_result,
            solubility=solubility_result,
            aggregation=aggregation_result,
            immunogenicity=immunogenicity_result,
            sequence=sequence,
            sequence_length=len(sequence),
            name=name,
            structure_provided=structure is not None,
            timestamp=datetime.utcnow(),
            weights_used=self._weights,
            recommendations=recommendations,
        )

        # Cache result
        if self._cache_enabled:
            self._cache[cache_key] = result

        logger.info(
            "Scored %s: %.1f/100 (%s risk)",
            name or f"seq_{len(sequence)}aa",
            total_score,
            result.risk_level.value,
        )

        return result

    def score_batch(
        self,
        sequences: list[str],
        structures: list[str | None] | None = None,
        names: list[str | None] | None = None,
        parallel: bool = True,
    ) -> list[ProteinScoreResult]:
        """
        Calculate developability scores for multiple proteins.

        Args:
            sequences: List of amino acid sequences.
            structures: Optional list of PDB structures (same length as sequences).
            names: Optional list of protein names (same length as sequences).
            parallel: Use parallel processing (default: True).

        Returns:
            List of ProteinScoreResult objects in the same order as input.

        Raises:
            InvalidSequenceError: If any sequence is invalid.
        """
        n = len(sequences)
        if structures is None:
            structures = [None] * n
        if names is None:
            names = [None] * n

        if len(structures) != n or len(names) != n:
            raise ValueError("sequences, structures, and names must have same length")

        if not parallel or n <= 2:
            # Sequential processing
            return [
                self.score(seq, struct, name)
                for seq, struct, name in zip(sequences, structures, names)
            ]

        # Parallel processing
        results: list[ProteinScoreResult | None] = [None] * n

        with ThreadPoolExecutor(max_workers=self._config.max_workers) as executor:
            futures = {
                executor.submit(self.score, seq, struct, name): i
                for i, (seq, struct, name) in enumerate(
                    zip(sequences, structures, names)
                )
            }

            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return [r for r in results if r is not None]

    def compare(
        self,
        results: list[ProteinScoreResult],
        output_format: str = "table",
    ) -> str | pd.DataFrame:
        """
        Compare multiple scoring results.

        Args:
            results: List of ProteinScoreResult objects.
            output_format: "table" (text), "dataframe" (pandas), or "json".

        Returns:
            Comparison table, DataFrame, or JSON string.
        """
        if not results:
            raise ValueError("No results to compare")

        data = []
        for r in results:
            data.append({
                "Name": r.name or f"seq_{r.sequence_length}aa",
                "Total": f"{r.total_score:.1f}",
                "Stability": f"{r.stability.score:.1f}",
                "Solubility": f"{r.solubility.score:.1f}",
                "Aggregation": f"{r.aggregation.score:.1f}",
                "Immunogenicity": f"{r.immunogenicity.score:.1f}",
                "Risk": r.risk_level.value.upper(),
            })

        if output_format == "dataframe":
            import pandas as pd
            return pd.DataFrame(data)

        if output_format == "json":
            import json
            return json.dumps(data, indent=2)

        # Default: text table
        return self._format_comparison_table(data)

    def _format_comparison_table(self, data: list[dict[str, str]]) -> str:
        """Format comparison data as text table."""
        headers = ["Name", "Total", "Stability", "Solubility", "Aggregation", "Immunogenicity", "Risk"]

        # Calculate column widths
        widths = {h: len(h) for h in headers}
        for row in data:
            for h in headers:
                widths[h] = max(widths[h], len(str(row.get(h, ""))))

        # Build table
        lines = []

        # Header
        header_line = " | ".join(h.ljust(widths[h]) for h in headers)
        lines.append(header_line)
        lines.append("-" * len(header_line))

        # Data rows
        for row in data:
            line = " | ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers)
            lines.append(line)

        return "\n".join(lines)

    def _generate_recommendations(
        self,
        stability: StabilityResult,
        solubility: SolubilityResult,
        aggregation: AggregationResult,
        immunogenicity: ImmunogenicityResult,
    ) -> list[Recommendation]:
        """Generate actionable recommendations based on component scores."""
        recommendations = []

        # Stability recommendations
        if stability.score < 40:
            recommendations.append(Recommendation(
                category="stability",
                severity="critical",
                message="Poor thermostability predicted. Consider introducing stabilizing mutations.",
                region=stability.unstable_regions[0] if stability.unstable_regions else None,
            ))
        elif stability.score < 60:
            recommendations.append(Recommendation(
                category="stability",
                severity="warning",
                message="Moderate stability. Consider optimizing unstable regions.",
            ))

        # Solubility recommendations
        if solubility.score < 40:
            recommendations.append(Recommendation(
                category="solubility",
                severity="critical",
                message="High aggregation risk during expression. Consider solubility-enhancing mutations.",
                region=solubility.insoluble_regions[0] if solubility.insoluble_regions else None,
            ))
        elif solubility.score < 60:
            recommendations.append(Recommendation(
                category="solubility",
                severity="warning",
                message="Moderate solubility. May require optimization for high-yield expression.",
            ))

        # Aggregation recommendations
        if aggregation.score < 40:
            recommendations.append(Recommendation(
                category="aggregation",
                severity="critical",
                message=f"High aggregation propensity ({aggregation.apr_count} APRs). Engineering strongly recommended.",
                region=aggregation.aggregation_prone_regions[0] if aggregation.aggregation_prone_regions else None,
            ))
        elif aggregation.score < 60:
            recommendations.append(Recommendation(
                category="aggregation",
                severity="warning",
                message=f"Moderate aggregation risk ({aggregation.apr_count} APRs). Monitor during development.",
            ))

        # Immunogenicity recommendations
        if immunogenicity.score < 40:
            recommendations.append(Recommendation(
                category="immunogenicity",
                severity="critical",
                message=f"High immunogenicity risk ({immunogenicity.strong_binders} strong epitopes). Deimmunization recommended.",
            ))
        elif immunogenicity.score < 60:
            recommendations.append(Recommendation(
                category="immunogenicity",
                severity="warning",
                message=f"Moderate immunogenicity ({immunogenicity.epitope_count} epitopes). Consider deimmunization for therapeutic use.",
            ))

        # Sort by severity
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        recommendations.sort(key=lambda r: severity_order.get(r.severity, 3))

        return recommendations

    def _get_cache_key(self, sequence: str, structure: str | None) -> str:
        """Generate cache key for a sequence."""
        import hashlib

        data = f"{sequence}:{structure or ''}:{self._weights}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._cache.clear()
        logger.info("Cache cleared")

    @property
    def weights(self) -> ScoringWeights:
        """Get current scoring weights."""
        return self._weights

    @weights.setter
    def weights(self, value: ScoringWeights) -> None:
        """Set scoring weights."""
        self._weights = value
        self.clear_cache()  # Invalidate cache when weights change

    def __repr__(self) -> str:
        return (
            f"ProteinScore(api_key={'set' if self._api_key else 'anonymous'}, "
            f"local_only={self._local_only})"
        )
