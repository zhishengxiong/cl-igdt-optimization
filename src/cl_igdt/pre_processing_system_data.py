import numpy as np
import pandas as pd


def build_load_factor(T):
    if T == 1:
        return np.array([1])

    load_factor = np.array([
        0.75, 0.73, 0.72, 0.76, 0.77, 0.78,
        0.8, 1.1, 1.25, 1.24, 1.23, 1.3,
        1.23, 1.19, 1.2, 1.21, 1.18, 1.17,
        1.1, 1.0, 0.9, 0.95, 0.8, 0.76
    ])

    return load_factor[:T]


def build_reference_voltage_vector(num_nodes):
    return np.ones((num_nodes - 1, 1)) * 12.66 * 12.66


def read_network_data(filename):
    raw_lines = pd.read_excel(filename, sheet_name="Lines")
    raw_nodes = pd.read_excel(filename, sheet_name="Nodes")

    return raw_lines, raw_nodes


def build_reduced_incidence_matrix(raw_lines, num_nodes):
    branch_end_nodes = {
        i + 1: (int(raw_lines.loc[i, "FROM"]), int(raw_lines.loc[i, "TO"]))
        for i in range(len(raw_lines))
    }

    A_full = np.zeros((len(branch_end_nodes), num_nodes))

    for branch, (start_node, end_node) in branch_end_nodes.items():
        A_full[branch - 1, start_node - 1] = 1
        A_full[branch - 1, end_node - 1] = -1

    A = np.delete(A_full, 0, axis=1)

    return A


def build_voltage_sensitivity_matrices(raw_lines, A_inv):
    R = raw_lines["R"].values
    R_diag = np.diag(R)
    R_prime = 2 * A_inv @ R_diag @ A_inv.T

    X = raw_lines["X"].values
    X_diag = np.diag(X)
    X_prime = 2 * A_inv @ X_diag @ A_inv.T

    return R_prime, X_prime


def build_load_profiles(raw_nodes, load_factor):
    PD = -raw_nodes["PD"].values
    PD = np.delete(PD, 0)
    P_load = np.outer(PD, load_factor)

    QD = -raw_nodes["QD"].values
    QD = np.delete(QD, 0)
    Q_load = np.outer(QD, load_factor)

    return P_load, Q_load


def pre_processing_system_data(filename, num_nodes, T):
    load_factor = build_load_factor(T)

    v0 = build_reference_voltage_vector(num_nodes)

    raw_lines, raw_nodes = read_network_data(filename)

    A = build_reduced_incidence_matrix(raw_lines, num_nodes)
    A_inv = np.linalg.inv(A)

    R_prime, X_prime = build_voltage_sensitivity_matrices(raw_lines, A_inv)

    P_load, Q_load = build_load_profiles(raw_nodes, load_factor)

    system_data = [v0, A, R_prime, X_prime, P_load, Q_load]

    return system_data