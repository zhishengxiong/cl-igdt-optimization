import numpy as np
import pandas as pd
import pytest

from cl_igdt import system_data_preprocessing as psd


def test_build_load_factor_for_single_period():
    np.testing.assert_array_equal(psd.build_load_factor(1), np.array([1]))


def test_build_load_factor_uses_first_t_values():
    np.testing.assert_array_equal(
        psd.build_load_factor(3),
        np.array([0.75, 0.73, 0.72]),
    )


def test_build_load_factor_rejects_t_beyond_profile_length():
    with pytest.raises(ValueError, match="exceeds the load factor profile length"):
        psd.build_load_factor(25)


def test_build_reference_voltage_vector():
    network_config = {"base_voltage_kv": 12.66}

    v0 = psd.build_reference_voltage_vector(num_nodes=3, network_config=network_config)

    assert v0.shape == (2, 1)
    np.testing.assert_allclose(v0, np.ones((2, 1)) * 12.66 * 12.66)


def test_read_network_data_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Network data file not found"):
        psd.read_network_data(tmp_path / "missing.xlsx")


def test_read_network_data_rejects_missing_sheet(tmp_path):
    network_file = tmp_path / "network.xlsx"
    with pd.ExcelWriter(network_file) as writer:
        pd.DataFrame({"x": [1]}).to_excel(writer, sheet_name=psd.LINES_SHEET, index=False)

    with pytest.raises(ValueError, match="Missing sheet 'Nodes'"):
        psd.read_network_data(network_file)


def test_build_system_data_for_three_bus_radial_network(tmp_path):
    network_file = tmp_path / "network.xlsx"

    raw_lines = pd.DataFrame({
        psd.FROM_COLUMN: [1, 2],
        psd.TO_COLUMN: [2, 3],
        psd.R_COLUMN: [0.1, 0.2],
        psd.X_COLUMN: [0.01, 0.02],
    })
    raw_nodes = pd.DataFrame({
        psd.PD_COLUMN: [0.0, 10.0, 20.0],
        psd.QD_COLUMN: [0.0, 1.0, 2.0],
    })

    with pd.ExcelWriter(network_file) as writer:
        raw_lines.to_excel(writer, sheet_name=psd.LINES_SHEET, index=False)
        raw_nodes.to_excel(writer, sheet_name=psd.NODES_SHEET, index=False)

    network_config = {"base_voltage_kv": 12.66, "slack_bus_index": 0}

    system_data = psd.build_system_data(
        network_file=network_file,
        num_nodes=3,
        T=2,
        network_config=network_config,
    )

    assert system_data.A.shape == (2, 2)
    assert system_data.R_prime.shape == (2, 2)
    assert system_data.X_prime.shape == (2, 2)
    assert system_data.P_load.shape == (2, 2)
    assert system_data.Q_load.shape == (2, 2)

    expected_p_load = np.outer(np.array([-10.0, -20.0]), np.array([0.75, 0.73]))
    expected_q_load = np.outer(np.array([-1.0, -2.0]), np.array([0.75, 0.73]))

    np.testing.assert_allclose(system_data.P_load, expected_p_load)
    np.testing.assert_allclose(system_data.Q_load, expected_q_load)
