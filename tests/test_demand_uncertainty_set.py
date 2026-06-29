import pandas as pd
import pytest

from cl_igdt import demand_uncertainty_set as demand


def test_read_historical_demand_data_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Historical demand data file not found"):
        demand.read_historical_demand_data(tmp_path, num_nodes=3)


def test_initialize_load_uncertainty_set_shape_and_keys():
    uset = demand.initialize_load_uncertainty_set(
        partition_num=3,
        iteration=1,
        alpha_ini=0,
        nodes=2,
        T=4,
    )

    assert list(uset.keys()) == [0.0, 0.1, 0.2]
    assert all(value.shape == (2, 4) for value in uset.values())


def test_extract_demand_samples_rejects_missing_t0(tmp_path):
    demand_file = tmp_path / "historical_data_demand_3.xlsx"
    pd.DataFrame({"time": ["t1"], "sample_1": [10.0]}).to_excel(
        demand_file, sheet_name="Bus1", index=False
    )

    with pytest.raises(ValueError, match="Missing t0 row in demand sheet: Bus1"):
        demand.extract_demand_samples(demand_file, "Bus1")


def test_cumulative_sum():
    assert demand.cumulative_sum([2, 3, 5]) == [2, 5, 10]
