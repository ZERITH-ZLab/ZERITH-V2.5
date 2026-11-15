"""Semantic action contracts shared by training and inference."""

from .absolute_pose import (  # noqa: F401
    ABSOLUTE_EEF_POSE_CONTRACT,
    ABSOLUTE_EEF_POSE_DIM,
    ABSOLUTE_EEF_POSE_LAYOUT,
    INTERNAL_ACTION_CONTRACT,
    INTERNAL_ACTION_DIM,
    INTERNAL_ACTION_LAYOUT,
    INTERNAL_ACTION_VALID_INDICES,
    MODEL_SIDE_NORMALIZATION_ID,
    POSE_CODEC_STATS_FORMAT,
    CanonicalAbsolutePoseCodec,
)
from .state_trajectory import (  # noqa: F401
    ACTION20_COORDINATE_CONVENTION,
    Action20State20Codec,
    INTERNAL_STATE_ACTION_CONTRACT,
    INTERNAL_STATE_ACTION_DIM,
    INTERNAL_STATE_ACTION_LAYOUT,
    MODEL_SIDE_STATE_NORMALIZATION_ID,
    STATE_CODEC_STATS_FORMAT,
    STATE_TRAJECTORY_CONTRACT,
    STATE_TRAJECTORY_DIM,
    STATE_TRAJECTORY_LAYOUT,
    CanonicalStateTrajectoryCodec,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from .libero_action20 import (  # noqa: F401
    LIBERO_STATE20_TARGET_SEMANTICS,
    LiberoAction20Adapter,
)
from .state20_targets import (  # noqa: F401
    LIBERO_STATE20_TARGET_CONTRACT_ID,
    ROBOT_STATE20_TARGET_CONTRACT_ID,
    ROBOT_STATE20_TARGET_SEMANTICS,
    STATE20_TARGET_CONTRACTS,
    state20_target_contract_registry,
)

from .schema import (  # noqa: F401
    ActionProfile,
    ActionSchema,
    build_action_schema,
    expand_action_feature_mask,
    resolve_action_profile,
    validate_action_stats,
)
