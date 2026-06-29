import numpy as np
import pandas as pd
import pytest

from cl_igdt import demand_uncertainty_set as demand
from cl_igdt import pv_uncertainty_set as pv


def test_extract_pv_samples_sorts_rounds_counts_and_requires_t0():
    raw_pv_data = pd.DataFrame(
        {
            "time": ["t0", "t1"],
            "s1": [2.04, 0.0],
            "s2": [1.04, 2.0],
            "s3": [1.04, 2.04],
        }
    )

    PV, PV_k, n_k, n = pv.extract_pv_samples(raw_pv_data)

    assert PV["t0"] == [1.0, 1.0, 2.0]
    assert PV_k["t0"] == [1.0, 2.0]
    assert n_k["t0"] == [2, 3]
    assert n == 3

    with pytest.raises(ValueError, match="Missing t0 row"):
        pv.extract_pv_samples(pd.DataFrame({"time": ["t1"], "s1": [1.0]}))


def test_shortest_interval_helpers_handle_zero_and_positive_confidence():
    values = np.array([0.0, 1.0, 3.0])
    cum_probs_upper = np.array([0.1, 0.9, 1.0])
    cum_probs_lower = np.array([0.0, 0.7, 1.0])

    assert pv.find_shortest_pv_interval(
        values,
        cum_probs_lower,
        cum_probs_upper,
        confidence_level=0,
    ) == (1.0, 1.0)

    assert pv.find_shortest_pv_interval(
        values,
        cum_probs_lower,
        np.array([0.4, 0.7, 1.0]),
        confidence_level=0.6,
    ) == (0, 1.0)

    assert demand.find_shortest_load_interval(
        np.array([0.0, 1.0, 2.0, 4.0]),
        np.array([0.0, 0.5, 0.85, 1.0]),
        np.array([0.1, 0.2, 0.4, 0.7]),
        confidence_level=0.6,
    ) == (1.0, 2.0)


def test_pv_uncertainty_set_initialization_and_tiling():
    pv_uset = pv.initialize_pv_uncertainty_set(
        partition_num=3,
        iteration=1,
        alpha_ini=0.2,
        T=2,
    )

    assert list(pv_uset) == [0.2, 0.3, 0.4]
    for confidence_level, values in pv_uset.items():
        values[:] = [confidence_level, confidence_level + 1]

    tiled = pv.tile_pv_uncertainty_set(
        pv_uset,
        partition_num=3,
        iteration=1,
        alpha_ini=0.2,
        num_pv=2,
    )

    np.testing.assert_allclose(tiled[0.3], np.array([[0.3, 1.3], [0.3, 1.3]]))
