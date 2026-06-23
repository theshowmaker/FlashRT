"""FlashRT — Action post-processing utilities."""

import numpy as np

LIBERO_ACTION_DIM = 7


def unnormalize_actions(actions, norm_stats):
    """Unnormalize actions using OpenPI's q01/q99 quantile formula.

    Match ``openpi.transforms.Unnormalize``: values outside the normalized
    ``[-1, 1]`` interval are linearly extrapolated instead of clipped.
    """
    q01 = np.array(norm_stats["actions"]["q01"], dtype=np.float32)
    q99 = np.array(norm_stats["actions"]["q99"], dtype=np.float32)
    dim = min(actions.shape[-1], len(q01))
    actions = np.asarray(actions, dtype=np.float32)
    unnorm = actions.copy()
    unnorm[..., :dim] = (
        (actions[..., :dim] + 1.0) / 2.0 * (q99[:dim] - q01[:dim] + 1e-6)
        + q01[:dim]
    )
    return unnorm
