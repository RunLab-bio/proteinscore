"""
Antibody Developability Scorer

Integrates TAP metrics, liability scanning, and ProteinScore for
comprehensive antibody developability assessment.

Targets SOTA performance on Jain 2017 / FLAb2 benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from proteinscore.antibody.cdr import CDRDetector, CDRRegions
from proteinscore.antibody.tap_metrics import TAPMetrics, TAPResult, TAPFlag
from proteinscore.antibody.liabilities import (
    LiabilityScanner,
    LiabilityScanResult,
    LiabilitySeverity,
)


# =============================================================================
# Result Structures
# =============================================================================

@dataclass
class AntibodyResult:
    """Complete antibody developability assessment."""
    # Sequences
    vh_sequence: str
    vl_sequence: str

    # TAP metrics
    tap_result: TAPResult

    # Liability analysis
    vh_liabilities: LiabilityScanResult
    vl_liabilities: LiabilityScanResult

    # Scores
    tap_score: float           # 0-100 based on TAP flags
    liability_score: float     # 0-100 based on liabilities
    hic_proxy: float           # HIC retention time proxy
    total_score: float         # Combined developability score

    # CDR info
    vh_cdrs: CDRRegions
    vl_cdrs: CDRRegions

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def risk_level(self) -> str:
        """Overall risk level."""
        if self.total_score >= 80:
            return "low"
        elif self.total_score >= 60:
            return "medium"
        elif self.total_score >= 40:
            return "high"
        else:
            return "critical"

    @property
    def n_red_flags(self) -> int:
        """Total number of red flags."""
        return self.tap_result.n_red_flags

    @property
    def n_high_severity_liabilities(self) -> int:
        """Total high-severity liabilities."""
        return (
            self.vh_liabilities.summary.high_severity +
            self.vl_liabilities.summary.high_severity
        )

    @property
    def total_liabilities(self) -> int:
        """Total number of liabilities."""
        return (
            self.vh_liabilities.summary.total +
            self.vl_liabilities.summary.total
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_score": self.total_score,
            "risk_level": self.risk_level,
            "tap_score": self.tap_score,
            "liability_score": self.liability_score,
            "hic_proxy": self.hic_proxy,
            "tap_metrics": {
                "cdr_length": self.tap_result.cdr_length.value,
                "psh": self.tap_result.psh.value,
                "ppc": self.tap_result.ppc.value,
                "pnc": self.tap_result.pnc.value,
                "sfvcsp": self.tap_result.sfvcsp.value,
            },
            "tap_flags": {
                "cdr_length": self.tap_result.cdr_length.flag.value,
                "psh": self.tap_result.psh.flag.value,
                "ppc": self.tap_result.ppc.flag.value,
                "pnc": self.tap_result.pnc.flag.value,
                "sfvcsp": self.tap_result.sfvcsp.flag.value,
            },
            "liabilities": {
                "vh_total": self.vh_liabilities.summary.total,
                "vl_total": self.vl_liabilities.summary.total,
                "high_severity": self.n_high_severity_liabilities,
                "deamidation": (
                    self.vh_liabilities.summary.deamidation +
                    self.vl_liabilities.summary.deamidation
                ),
                "isomerization": (
                    self.vh_liabilities.summary.isomerization +
                    self.vl_liabilities.summary.isomerization
                ),
                "oxidation": (
                    self.vh_liabilities.summary.oxidation +
                    self.vl_liabilities.summary.oxidation
                ),
            },
            "vh_charge": self.tap_result.vh_charge,
            "vl_charge": self.tap_result.vl_charge,
            "cdr_sequences": self.tap_result.cdr_sequences,
        }


# =============================================================================
# Antibody Scorer
# =============================================================================

class AntibodyScorer:
    """
    Comprehensive antibody developability scorer.

    Combines:
    - TAP metrics (CDR length, PSH, PPC, PNC, SFvCSP)
    - Liability scanning (deamidation, isomerization, etc.)
    - HIC retention proxy (Jain hydrophobicity scale)

    Targets SOTA performance on Jain 2017 / FLAb2 benchmarks.
    """

    def __init__(
        self,
        hydrophobicity_scale: str = "jain",
        check_aggregation: bool = True,
    ):
        """
        Initialize antibody scorer.

        Args:
            hydrophobicity_scale: Scale for PSH calculation
                                 "jain" recommended for HIC correlation
            check_aggregation: Also scan for aggregation-prone regions
        """
        self._cdr_detector = CDRDetector(numbering_scheme="chothia")
        self._tap_metrics = TAPMetrics(
            hydrophobicity_scale=hydrophobicity_scale,
            cdr_detector=self._cdr_detector,
        )
        self._liability_scanner = LiabilityScanner(
            cdr_detector=self._cdr_detector,
            check_aggregation=check_aggregation,
        )

    def score(
        self,
        vh_sequence: str,
        vl_sequence: str,
    ) -> AntibodyResult:
        """
        Score an antibody for developability.

        Args:
            vh_sequence: Heavy chain variable region sequence
            vl_sequence: Light chain variable region sequence

        Returns:
            AntibodyResult with comprehensive developability assessment
        """
        vh_sequence = vh_sequence.upper().strip()
        vl_sequence = vl_sequence.upper().strip()

        # Calculate TAP metrics
        tap_result = self._tap_metrics.calculate(vh_sequence, vl_sequence)

        # Scan for liabilities
        vh_liabilities = self._liability_scanner.scan(vh_sequence, "heavy")
        vl_liabilities = self._liability_scanner.scan(vl_sequence, "light")

        # Calculate HIC proxy
        hic_proxy = self._tap_metrics.calculate_hic_proxy(vh_sequence, vl_sequence)

        # Detect CDRs
        vh_cdrs, vl_cdrs = self._cdr_detector.detect_paired(vh_sequence, vl_sequence)

        # Calculate component scores
        tap_score = tap_result.developability_score
        liability_score = (vh_liabilities.liability_score + vl_liabilities.liability_score) / 2

        # Combined score (weighted)
        # TAP: 50%, Liabilities: 30%, HIC normalization: 20%
        # HIC proxy typically ranges from -0.5 to 0.5
        # Convert to 0-100 scale (lower hydrophobicity = better)
        hic_score = max(0, min(100, 50 - (hic_proxy * 100)))

        total_score = (
            0.50 * tap_score +
            0.30 * liability_score +
            0.20 * hic_score
        )

        return AntibodyResult(
            vh_sequence=vh_sequence,
            vl_sequence=vl_sequence,
            tap_result=tap_result,
            vh_liabilities=vh_liabilities,
            vl_liabilities=vl_liabilities,
            tap_score=round(tap_score, 1),
            liability_score=round(liability_score, 1),
            hic_proxy=hic_proxy,
            total_score=round(total_score, 1),
            vh_cdrs=vh_cdrs,
            vl_cdrs=vl_cdrs,
        )

    def score_batch(
        self,
        vh_sequences: list[str],
        vl_sequences: list[str],
    ) -> list[AntibodyResult]:
        """
        Score multiple antibodies.

        Args:
            vh_sequences: List of VH sequences
            vl_sequences: List of VL sequences (same length as vh_sequences)

        Returns:
            List of AntibodyResult objects
        """
        if len(vh_sequences) != len(vl_sequences):
            raise ValueError("vh_sequences and vl_sequences must have same length")

        return [
            self.score(vh, vl)
            for vh, vl in zip(vh_sequences, vl_sequences)
        ]

    def get_hic_score(
        self,
        vh_sequence: str,
        vl_sequence: str,
    ) -> float:
        """
        Get HIC retention time proxy score.

        This correlates with experimental HIC data.
        Target: Spearman ρ > 0.5 on Jain 2017 dataset.

        Args:
            vh_sequence: VH sequence
            vl_sequence: VL sequence

        Returns:
            HIC proxy score (higher = more hydrophobic)
        """
        return self._tap_metrics.calculate_hic_proxy(vh_sequence, vl_sequence)

    def get_aggregation_score(
        self,
        vh_sequence: str,
        vl_sequence: str,
    ) -> float:
        """
        Get aggregation propensity score.

        Combines PSH (surface hydrophobicity) with liability count.
        Higher score = lower aggregation risk.

        Args:
            vh_sequence: VH sequence
            vl_sequence: VL sequence

        Returns:
            Aggregation score (0-100, higher = better)
        """
        tap_result = self._tap_metrics.calculate(vh_sequence, vl_sequence)

        # PSH contributes to aggregation risk
        # Lower PSH = better (less hydrophobic patches)
        psh_score = 100 - min(100, tap_result.psh.value / 2)

        # Liability count also affects aggregation
        vh_liabilities = self._liability_scanner.scan(vh_sequence, "heavy")
        vl_liabilities = self._liability_scanner.scan(vl_sequence, "light")

        total_apr = (
            vh_liabilities.summary.aggregation +
            vl_liabilities.summary.aggregation
        )
        apr_penalty = min(30, total_apr * 5)

        return max(0, psh_score - apr_penalty)

    def get_expression_score(
        self,
        vh_sequence: str,
        vl_sequence: str,
    ) -> float:
        """
        Get predicted expression score.

        Based on:
        - Charge balance (SFvCSP)
        - CDR length
        - Liability count

        Args:
            vh_sequence: VH sequence
            vl_sequence: VL sequence

        Returns:
            Expression score (0-100, higher = better predicted expression)
        """
        tap_result = self._tap_metrics.calculate(vh_sequence, vl_sequence)

        # Start with 80
        score = 80.0

        # CDR length penalty (too short or too long)
        cdr_len = tap_result.cdr_length.value
        if cdr_len < 40 or cdr_len > 60:
            score -= 10
        if cdr_len < 35 or cdr_len > 70:
            score -= 10

        # SFvCSP penalty (charge asymmetry)
        if tap_result.sfvcsp.flag == TAPFlag.RED:
            score -= 15
        elif tap_result.sfvcsp.flag == TAPFlag.AMBER:
            score -= 5

        # Liability penalties
        vh_liabilities = self._liability_scanner.scan(vh_sequence, "heavy")
        vl_liabilities = self._liability_scanner.scan(vl_sequence, "light")

        # Unpaired cysteines severely affect expression
        unpaired = (
            vh_liabilities.summary.unpaired_cysteine +
            vl_liabilities.summary.unpaired_cysteine
        )
        score -= unpaired * 20

        # High severity liabilities
        high_sev = (
            vh_liabilities.summary.high_severity +
            vl_liabilities.summary.high_severity
        )
        score -= high_sev * 3

        return max(0, min(100, score))

    def print_report(self, result: AntibodyResult) -> str:
        """
        Generate a human-readable report.

        Args:
            result: AntibodyResult from score()

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("ANTIBODY DEVELOPABILITY REPORT")
        lines.append("=" * 60)

        lines.append(f"\nTotal Score: {result.total_score}/100 ({result.risk_level.upper()} RISK)")
        lines.append(f"TAP Score: {result.tap_score}/100")
        lines.append(f"Liability Score: {result.liability_score}/100")
        lines.append(f"HIC Proxy: {result.hic_proxy:.3f}")

        lines.append("\n" + "-" * 60)
        lines.append("TAP METRICS")
        lines.append("-" * 60)

        for metric in result.tap_result.all_metrics:
            flag_symbol = {"green": "✓", "amber": "⚠", "red": "✗"}[metric.flag.value]
            lines.append(f"  {metric.name:<15} {metric.value:>8.1f}  {flag_symbol} {metric.flag.value.upper()}")

        lines.append("\n" + "-" * 60)
        lines.append("CHARGE ANALYSIS")
        lines.append("-" * 60)
        lines.append(f"  VH Charge: {result.tap_result.vh_charge:+.1f}")
        lines.append(f"  VL Charge: {result.tap_result.vl_charge:+.1f}")

        lines.append("\n" + "-" * 60)
        lines.append("LIABILITIES")
        lines.append("-" * 60)

        vh_sum = result.vh_liabilities.summary
        vl_sum = result.vl_liabilities.summary

        lines.append(f"  VH: {vh_sum.total} total ({vh_sum.high_severity} high severity)")
        lines.append(f"    Deamidation: {vh_sum.deamidation}, Isomerization: {vh_sum.isomerization}")
        lines.append(f"    Oxidation: {vh_sum.oxidation}, Glycosylation: {vh_sum.glycosylation}")

        lines.append(f"  VL: {vl_sum.total} total ({vl_sum.high_severity} high severity)")
        lines.append(f"    Deamidation: {vl_sum.deamidation}, Isomerization: {vl_sum.isomerization}")
        lines.append(f"    Oxidation: {vl_sum.oxidation}, Glycosylation: {vl_sum.glycosylation}")

        # Top recommendations
        if result.n_red_flags > 0 or result.n_high_severity_liabilities > 0:
            lines.append("\n" + "-" * 60)
            lines.append("TOP RECOMMENDATIONS")
            lines.append("-" * 60)

            # TAP red flags
            for metric in result.tap_result.all_metrics:
                if metric.flag == TAPFlag.RED:
                    lines.append(f"  ✗ {metric.name} is out of range ({metric.value:.1f})")

            # High severity liabilities
            all_liabilities = (
                result.vh_liabilities.liabilities +
                result.vl_liabilities.liabilities
            )
            high_sev = [l for l in all_liabilities if l.severity == LiabilitySeverity.HIGH]

            for lib in high_sev[:5]:  # Top 5
                lines.append(f"  ✗ {lib.type.value} at position {lib.position+1}: {lib.motif}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)
