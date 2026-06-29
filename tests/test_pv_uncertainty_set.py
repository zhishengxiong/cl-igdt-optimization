import numpy as np
import pandas as pd
import pytest

from cl_igdt import pv_uncertainty_set as pv


def test_read_historical_pv_data_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Historical PV data file not found"):
        pv.read_historical_pv_data(tmp_path, num_nodes=3)


def test_read_historical_pv_data_rejects_missing_sheet(tmp_path):
    pv_file = tmp_path / "historical_data_PV_3.xlsx"
    pd.DataFrame({"x": [1]}).to_excel(pv_file, sheet_name="WrongSheet", index=False)

    with pytest.raises(ValueError, match="Missing sheet 'PVData'"):
        pv.read_historical_pv_data(tmp_path, num_nodes=3)


def test_initialize_pv_uncertainty_set_shape_and_keys():
    uset = pv.initialize_pv_uncertainty_set(
        partition_num=3,
        iteration=1,
        alpha_ini=0,
        T=4,
    )

    assert list(uset.keys()) == [0.0, 0.1, 0.2]
    assert all(value.shape == (4,) for value in uset.values())


def test_extract_pv_samples_rejects_missing_t0():
    raw_pv_data = pd.DataFrame({"time": ["t1"], "sample_1": [2.0]})

    with pytest.raises(ValueError, match="Missing t0 row in sheet: PVData"):
        pv.extract_pv_samples(raw_pv_data)


def test_tile_pv_uncertainty_set():
    uset = {0.0: np.array([1.0, 2.0]), 0.1: np.array([3.0, 4.0])}

    tiled = pv.tile_pv_uncertainty_set(
        pv_uset=uset,
        partition_num=2,
        iteration=1,
        alpha_ini=0,
        num_pv=3,
    )

    assert tiled[0.0].shape == (3, 2)
    np.testing.assert_array_equal(
        tiled[0.0],
        np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]),
    )
