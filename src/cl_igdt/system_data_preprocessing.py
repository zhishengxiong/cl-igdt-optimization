"""Build network data for the optimization model.

This module converts network inputs into incidence matrices,
network parameter matrices, and time-dependent load profiles.
"""


from dataclasses import dataclass

import numpy as np
import pandas as pd


LINES_SHEET = "Lines"
NODES_SHEET = "Nodes"

FROM_COLUMN = "FROM"
TO_COLUMN = "TO"

R_COLUMN = "R"
X_COLUMN = "X"
PD_COLUMN = "PD"
QD_COLUMN = "QD"

LOAD_FACTOR_PROFILE = np.array([
    0.75, 0.73, 0.72, 0.76, 0.77, 0.78,
    0.8, 1.1, 1.25, 1.24, 1.23, 1.3,
    1.23, 1.19, 1.2, 1.21, 1.18, 1.17,
    1.1, 1.0, 0.9, 0.95, 0.8, 0.76,
])


@dataclass
class SystemData:
    v0: np.ndarray
    A: np.ndarray
    R_prime: np.ndarray
    X_prime: np.ndarray
    P_load: np.ndarray
    Q_load: np.ndarray


def build_load_factor(T):
    if T == 1:
        return np.array([1])

    if T > len(LOAD_FACTOR_PROFILE):
        raise ValueError(
            f"T={T} exceeds the load factor profile length: {len(LOAD_FACTOR_PROFILE)}"
        )

    return LOAD_FACTOR_PROFILE[:T]


def build_reference_voltage_vector(num_nodes, network_config):
    base_voltage_kv = network_config["base_voltage_kv"]

    v0 = np.ones((num_nodes - 1, 1)) * base_voltage_kv * base_voltage_kv

    return v0


def read_network_data(network_file):
    if not network_file.exists():
        raise FileNotFoundError(f"Network data file not found: {network_file}")

    available_sheets = pd.ExcelFile(network_file).sheet_names

    for sheet in [LINES_SHEET, NODES_SHEET]:
        if sheet not in available_sheets:
            raise ValueError(f"Missing sheet '{sheet}' in network data file: {network_file}")

    raw_lines = pd.read_excel(network_file, sheet_name=LINES_SHEET)
    raw_nodes = pd.read_excel(network_file, sheet_name=NODES_SHEET)

    return raw_lines, raw_nodes


def build_reduced_incidence_matrix(raw_lines, num_nodes, network_config):
    slack_bus_index = network_config["slack_bus_index"]

    line_connections = {
        i + 1: (int(raw_lines.loc[i, FROM_COLUMN]), int(raw_lines.loc[i, TO_COLUMN]))
        for i in range(len(raw_lines))
    }

    A_full = np.zeros((len(line_connections), num_nodes))

    for branch, (start_node, end_node) in line_connections.items():
        A_full[branch - 1, start_node - 1] = 1
        A_full[branch - 1, end_node - 1] = -1

    A = np.delete(A_full, slack_bus_index, axis=1)

    return A


def build_voltage_sensitivity_matrices(raw_lines, A_inv):
    R = raw_lines[R_COLUMN].to_numpy()
    X = raw_lines[X_COLUMN].to_numpy()

    R_diag = np.diag(R)
    X_diag = np.diag(X)

    R_prime = 2 * A_inv @ R_diag @ A_inv.T
    X_prime = 2 * A_inv @ X_diag @ A_inv.T

    return R_prime, X_prime


def build_load_profiles(raw_nodes, load_factor, network_config):
    slack_bus_index = network_config["slack_bus_index"]

    PD = -raw_nodes[PD_COLUMN].to_numpy()
    QD = -raw_nodes[QD_COLUMN].to_numpy()

    PD = np.delete(PD, slack_bus_index)
    QD = np.delete(QD, slack_bus_index)

    P_load = np.outer(PD, load_factor)
    Q_load = np.outer(QD, load_factor)

    return P_load, Q_load


def build_system_data(network_file, num_nodes, T, network_config):
    load_factor = build_load_factor(T)

    v0 = build_reference_voltage_vector(num_nodes, network_config)

    raw_lines, raw_nodes = read_network_data(network_file)

    A = build_reduced_incidence_matrix(raw_lines, num_nodes, network_config)
    A_inv = np.linalg.inv(A)

    R_prime, X_prime = build_voltage_sensitivity_matrices(raw_lines, A_inv)

    P_load, Q_load = build_load_profiles(raw_nodes, load_factor, network_config)

    system_data = SystemData(
        v0=v0,
        A=A,
        R_prime=R_prime,
        X_prime=X_prime,
        P_load=P_load,
        Q_load=Q_load,
    )

    return system_data