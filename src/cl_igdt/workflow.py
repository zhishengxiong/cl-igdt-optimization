"""
Coordinate the full CL-IGDT workflow.

This module connects configuration loading, data preprocessing,
IDM-based uncertainty-set construction, and iterative economic dispatch solving.
"""

import numpy as np
import yaml

from cl_igdt import demand_uncertainty_set as demand
from cl_igdt import ders_data_preprocessing as pder
from cl_igdt import economic_dispatch_igdt as ed
from cl_igdt import pv_uncertainty_set as pv
from cl_igdt import system_data_preprocessing as psd

REQUIRED_CONFIG_SECTIONS = ["time", "files", "network", "optimization"]


def load_config(config_file):
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Config file is empty: {config_file}")

    for section in REQUIRED_CONFIG_SECTIONS:
        if section not in config:
            raise KeyError(f"Missing config section: {section}")

    return config


def load_input_data(data_dir, T, num_nodes, config):
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    network_file = data_dir / config["files"]["network_file"]
    ders_file = data_dir / config["files"]["ders_file"]

    system_data = psd.build_system_data(
        network_file,
        num_nodes,
        T,
        config["network"],
    )

    ders_data = pder.build_ders_data(ders_file, T)

    return system_data, ders_data


def load_uncertainty_sets(data_dir, num_nodes, iteration, partition_num, T, alpha_ini, num_pv):
    load_uncertainty_file = data_dir / f"Bus{num_nodes}_P_load_Uset_{alpha_ini}.npz"
    pv_uncertainty_file = data_dir / f"Bus{num_nodes}_PV_Uset_{alpha_ini}.npz"

    # Uncertainty sets are cached because IDM construction is deterministic
    # for the same data, partition, iteration, and alpha_ini.
    try:
        with np.load(load_uncertainty_file) as data:
            P_load_Uset = {float(key): data[key] for key in data}
    except FileNotFoundError:
        P_load_Uset = demand.build_demand_uncertainty_set(
            data_dir,
            num_nodes,
            iteration,
            partition_num,
            T,
            alpha_ini,
            verbose=False,
        )

    try:
        with np.load(pv_uncertainty_file) as data:
            PV_Uset = {float(key): data[key] for key in data}
    except FileNotFoundError:
        PV_Uset = pv.build_pv_uncertainty_set(
            data_dir,
            num_nodes,
            iteration,
            partition_num,
            T,
            alpha_ini,
            num_pv,
            verbose=False,
        )

    return P_load_Uset, PV_Uset


def solve_optimization_iteration(
    system_data,
    ders_data,
    num_nodes,
    T,
    P_load_Uset,
    PV_Uset,
    iteration,
    partition_num,
    alpha_ini,
    accuracy,
    network_config,
    optimization_config,
):
    optimal_result = ed.solve_economic_dispatch_igdt(
        system_data,
        ders_data,
        num_nodes,
        T,
        P_load_Uset,
        PV_Uset,
        iteration,
        partition_num,
        alpha_ini,
        network_config,
        optimization_config,
    )

    if optimal_result is None:
        raise RuntimeError(f"Optimization failed at iteration {iteration}.")

    theta_u = optimal_result.theta_u

    current_alpha = alpha_ini
    alpha_ini = round(
        np.nonzero(theta_u)[0][0] * 10 ** (-iteration) + current_alpha,
        accuracy,
    )

    return alpha_ini


def run_case(project_root):
    """Run one configured CL-IGDT case and return the optimal alpha."""
    data_dir = project_root / "data"
    config_file = project_root / "configs" / "config.yaml"

    config = load_config(config_file)

    T = config["time"]["T"]

    network_config = config["network"]
    optimization_config = config["optimization"]

    num_nodes = network_config["num_nodes"]
    accuracy = optimization_config["accuracy"]
    partition_num = optimization_config["partition_num"]
    alpha_ini = 0

    system_data, ders_data = load_input_data(
        data_dir,
        T,
        num_nodes,
        config,
    )

    for iteration in range(1, accuracy + 1):
        P_load_Uset, PV_Uset = load_uncertainty_sets(
            data_dir,
            num_nodes,
            iteration,
            partition_num,
            T,
            alpha_ini,
            len(ders_data.PV_node),
        )

        alpha_ini = solve_optimization_iteration(
            system_data,
            ders_data,
            num_nodes,
            T,
            P_load_Uset,
            PV_Uset,
            iteration,
            partition_num,
            alpha_ini,
            accuracy,
            network_config,
            optimization_config,
        )

        print(f"The optimal confidence level of iteration {iteration} is: {alpha_ini}")

    alpha = alpha_ini
    return alpha
