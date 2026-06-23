import numpy as np

from flash_rt.core.utils.actions import unnormalize_actions


def test_unnormalize_actions_matches_openpi_quantile_extrapolation():
    norm_stats = {
        "actions": {
            "q01": [-2.0, 10.0],
            "q99": [2.0, 20.0],
        }
    }
    actions = np.array(
        [
            [-1.5, 1.5, 7.0],
            [-1.0, 1.0, 8.0],
        ],
        dtype=np.float32,
    )

    out = unnormalize_actions(actions, norm_stats)

    expected = actions.copy()
    q01 = np.array(norm_stats["actions"]["q01"], dtype=np.float32)
    q99 = np.array(norm_stats["actions"]["q99"], dtype=np.float32)
    expected[..., :2] = (actions[..., :2] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    np.testing.assert_allclose(out, expected, rtol=0.0, atol=1e-6)

    assert out[0, 0] < q01[0]
    assert out[0, 1] > q99[1]
