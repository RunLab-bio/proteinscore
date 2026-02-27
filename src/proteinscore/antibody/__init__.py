"""
Antibody-Specific Developability Analysis

Comprehensive antibody sequence analysis for therapeutic development:

Features:
- CDR detection (Chothia, Kabat, IMGT numbering)
- TAP metrics (CDR length, PSH, PPC, PNC, SFvCSP)
- Liability detection (10+ liability types with severity scoring)
- Actionable engineering recommendations

References:
    - Raybould MIJ et al. (2019). "Five computational developability guidelines
      for therapeutic antibody profiling." PNAS 116(10):4025-4030
    - Khetan et al. (2024). "LAP: Liability Antibody Profiler." PLOS Comp Bio
    - Lu et al. (2019). "Deamidation and isomerization liability analysis of
      131 clinical-stage antibodies." mAbs
"""

from proteinscore.antibody.cdr import CDRDetector, CDRRegions
from proteinscore.antibody.tap_metrics import TAPMetrics, TAPResult, TAPFlag
from proteinscore.antibody.liabilities import (
    LiabilityScanner,
    Liability,
    LiabilityScanResult,
    LiabilityType,
    LiabilitySeverity,
    scan_antibody,
    get_engineering_recommendations,
)
from proteinscore.antibody.hydrophobicity import (
    HydrophobicityScale,
    HydrophobicPatch,
    HICPrediction,
    SelfAssociationPrediction,
    HydrophobicityAnalysis,
    predict_hic_retention,
    predict_self_association,
    analyze_hydrophobicity,
    analyze_antibody_hydrophobicity,
    detect_hydrophobic_patches,
    get_combined_hic_score,
    predict_hic_ml,
)
from proteinscore.antibody.hic_predictor import (
    HICPredictor,
    predict_hic,
    extract_all_features,
    get_feature_names,
)
from proteinscore.antibody.scorer import AntibodyScorer, AntibodyResult

__all__ = [
    # CDR Detection
    "CDRDetector",
    "CDRRegions",
    # TAP Metrics
    "TAPMetrics",
    "TAPResult",
    "TAPFlag",
    # Liability Detection
    "LiabilityScanner",
    "Liability",
    "LiabilityScanResult",
    "LiabilityType",
    "LiabilitySeverity",
    "scan_antibody",
    "get_engineering_recommendations",
    # Hydrophobicity Analysis (HIC/AC-SINS prediction)
    "HydrophobicityScale",
    "HydrophobicPatch",
    "HICPrediction",
    "SelfAssociationPrediction",
    "HydrophobicityAnalysis",
    "predict_hic_retention",
    "predict_self_association",
    "analyze_hydrophobicity",
    "analyze_antibody_hydrophobicity",
    "detect_hydrophobic_patches",
    "get_combined_hic_score",
    "predict_hic_ml",
    # ML-based HIC Predictor (GBM + 59 handcrafted features, ρ = 0.55)
    "HICPredictor",
    "predict_hic",
    "extract_all_features",
    "get_feature_names",
    # Full Scorer
    "AntibodyScorer",
    "AntibodyResult",
]
