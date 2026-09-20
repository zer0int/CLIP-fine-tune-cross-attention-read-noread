from .synthetic import PatchAlignedSyntheticBank, SceneSpec
from .core import load_model, VisualConditionRunner, preprocess_pil_batch
from .analysis import (
    DeltaCollector, randomized_svd_rows, principal_angles,
    rank_k_template, project_last_dim, displacement_metrics,
    b13_head_decomposition, pathway_decomposition,
)
from .jacobian import top_input_singular_directions

__all__ = [
    "PatchAlignedSyntheticBank", "SceneSpec", "load_model",
    "VisualConditionRunner", "preprocess_pil_batch", "DeltaCollector",
    "randomized_svd_rows", "principal_angles", "rank_k_template",
    "project_last_dim", "displacement_metrics", "b13_head_decomposition",
    "pathway_decomposition", "top_input_singular_directions",
]
