from collections import Counter

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import beta


IDM_CONFIDENCE_LEVEL = 0.95
IDM_EQUIVALENT_SAMPLE_SIZE = 1

DEMAND_LOWER_EXTENSION_RATIO = 0.5
INTERPOLATION_KIND = "linear"
ZERO_VALUE = 0
FULL_PROBABILITY = 1


def read_historical_demand_data(data_dir, num_nodes):
    demand_data_file = data_dir / f"historical_data_demand_{num_nodes}.xlsx"
    demand_sheets = pd.ExcelFile(demand_data_file).sheet_names

    return demand_data_file, demand_sheets


def build_idm_parameters():
    gamma = IDM_CONFIDENCE_LEVEL
    confidence_lower = (1 - gamma) / 2
    confidence_upper = (1 + gamma) / 2
    s = IDM_EQUIVALENT_SAMPLE_SIZE

    return confidence_lower, confidence_upper, s


def initialize_load_uncertainty_set(partition_num, iteration, alpha_ini, nodes, T):
    P_load_Uset = {
        round(10 ** (-iteration) * partition + alpha_ini, iteration): np.zeros((nodes, T))
        for partition in range(partition_num)
    }

    return P_load_Uset


def cumulative_sum(number):
    result = []
    current_sum = 0

    for element in number:
        current_sum += element
        result.append(current_sum)

    return result


def extract_demand_samples(demand_data_file, sheet):
    raw_demand_data = pd.read_excel(demand_data_file, sheet_name=sheet)

    D = {}
    D_k = {}

    for _, row in raw_demand_data.iterrows():
        D[row.iloc[0]] = [round(num, 1) for num in sorted(row.iloc[1:].tolist())]
        D_k[row.iloc[0]] = sorted(set(D[row.iloc[0]]))

    value_counts = {key: Counter(values) for key, values in D.items()}
    sample_counts = {key: list(element.values()) for key, element in value_counts.items()}

    n_k = {key: cumulative_sum(values) for key, values in sample_counts.items()}
    n = len(D["t0"])

    return D, D_k, n_k, n


def build_idm_interpolated_cdf(D_k, n_k, n, confidence_lower, confidence_upper, s):
    icdf_lower = {}
    icdf_upper = {}

    final_D_k = {}
    final_icdf_lower = {}
    final_icdf_upper = {}

    for t, n_k_values in n_k.items():
        if D_k[t][0] > ZERO_VALUE:
            D_min_new = round(D_k[t][0] * DEMAND_LOWER_EXTENSION_RATIO, 2)
            D_k[t].insert(0, max(ZERO_VALUE, D_min_new))

            icdf_lower[t] = [
                beta.ppf(confidence_lower, i, s + n - i)
                for i in n_k_values
            ]
            icdf_lower[t].insert(0, ZERO_VALUE)

            n_k_values = n_k_values[:-1]
            n_k_values.insert(0, ZERO_VALUE)
            icdf_upper[t] = [
                beta.ppf(confidence_upper, s + i, n - i)
                for i in n_k_values
            ]
            icdf_upper[t].append(FULL_PROBABILITY)

        elif D_k[t][-1] > ZERO_VALUE:
            icdf_lower[t] = [
                beta.ppf(confidence_lower, i, s + n - i)
                for i in n_k_values
            ]

            n_k_values = n_k_values[:-1]
            icdf_upper[t] = [
                beta.ppf(confidence_upper, s + i, n - i)
                for i in n_k_values
            ]
            icdf_upper[t].append(FULL_PROBABILITY)

        else:
            icdf_lower[t] = [FULL_PROBABILITY]
            icdf_upper[t] = [FULL_PROBABILITY]

        if len(D_k[t]) >= 2:
            f_inter_lower = interp1d(D_k[t], icdf_lower[t], kind=INTERPOLATION_KIND)
            f_inter_upper = interp1d(D_k[t], icdf_upper[t], kind=INTERPOLATION_KIND)

            x_z = [
                (D_k[t][i] + D_k[t][i + 1]) * 0.5
                for i in range(len(D_k[t]) - 1)
            ]
            x_z = [round(num, 2) for num in x_z]

            final_D_k[t] = sorted(D_k[t] + x_z)
            final_icdf_lower[t] = f_inter_lower(final_D_k[t])
            final_icdf_upper[t] = f_inter_upper(final_D_k[t])

        else:
            final_D_k[t] = D_k[t]
            final_icdf_lower[t] = icdf_lower[t]
            final_icdf_upper[t] = icdf_upper[t]

    return final_D_k, final_icdf_lower, final_icdf_upper


def initialize_confidence_interval(confidence_level, confidence_interval, T):
    if confidence_level not in confidence_interval:
        confidence_interval[confidence_level] = {
            f"t{i}": (ZERO_VALUE, ZERO_VALUE) for i in range(T)
        }

    return confidence_interval


def find_shortest_load_interval(values, cum_probs_lower, cum_probs_upper, confidence_level):
    prob_densities_upper = np.diff(cum_probs_upper) / np.diff(values)
    prob_densities_upper = np.insert(
        prob_densities_upper,
        0,
        cum_probs_upper[0]
    )

    max_density_idx_upper = np.argmax(prob_densities_upper)

    shortest_interval = (np.min(values), np.max(values))
    min_width = np.max(values) - np.min(values)

    if confidence_level == ZERO_VALUE:
        shortest_interval = (
            values[max_density_idx_upper],
            values[max_density_idx_upper],
        )
    else:
        for i in range(len(values) - 1):
            for j in range(i + 1, len(values)):
                if cum_probs_lower[j] - cum_probs_upper[i] >= confidence_level:
                    width = values[j] - values[i]
                    if width < min_width:
                        min_width = width
                        shortest_interval = (values[i], values[j])
                    break

    return shortest_interval


def build_shortest_load_confidence_intervals(
        final_D_k, final_icdf_lower, final_icdf_upper,
        partition_num, iteration, alpha_ini,
        P_load_Uset, node_index, T
):
    confidence_interval = {}

    for t, _ in final_D_k.items():
        values = np.array(final_D_k[t])
        cum_probs_lower = final_icdf_lower[t]
        cum_probs_upper = final_icdf_upper[t]

        for partition in range(partition_num):
            confidence_level = round(10 ** (-iteration) * partition + alpha_ini, iteration)

            confidence_interval = initialize_confidence_interval(
                confidence_level, confidence_interval, T
            )

            shortest_interval = find_shortest_load_interval(
                values, cum_probs_lower, cum_probs_upper, confidence_level
            )

            P_load_Uset[confidence_level][node_index, int(t[1:])] = -round(
                shortest_interval[1], 2
            )
            confidence_interval[confidence_level][t] = shortest_interval

    return P_load_Uset, confidence_interval


def print_load_confidence_intervals(partition_num, iteration, alpha_ini, node_index, confidence_interval):
    for partition in range(partition_num):
        confidence_level = round(10 ** (-iteration) * partition + alpha_ini, iteration)
        print(
            f"The shortest CI of Demand of node {node_index + 1}, "
            f"confidence level {confidence_level} is:",
            confidence_interval[confidence_level],
        )


def save_load_uncertainty_set(P_load_Uset, data_dir, num_nodes, alpha_ini):
    output_file = data_dir / f"Bus{num_nodes}_P_load_Uset_{alpha_ini}.npz"

    np.savez(
        output_file,
        **{str(key): value for key, value in P_load_Uset.items()}
    )


def build_demand_uncertainty_set(
        data_dir, num_nodes, iteration, partition_num, T, alpha_ini,
        verbose=False
):
    demand_data_file, demand_sheets = read_historical_demand_data(data_dir, num_nodes)

    nodes = len(demand_sheets)

    confidence_lower, confidence_upper, s = build_idm_parameters()

    P_load_Uset = initialize_load_uncertainty_set(
        partition_num, iteration, alpha_ini, nodes, T
    )

    for sheet in demand_sheets:
        node_index = demand_sheets.index(sheet)

        D, D_k, n_k, n = extract_demand_samples(demand_data_file, sheet)

        final_D_k, final_icdf_lower, final_icdf_upper = build_idm_interpolated_cdf(
            D_k, n_k, n,
            confidence_lower, confidence_upper, s
        )

        P_load_Uset, confidence_interval = build_shortest_load_confidence_intervals(
            final_D_k, final_icdf_lower, final_icdf_upper,
            partition_num, iteration, alpha_ini,
            P_load_Uset, node_index, T
        )

        if verbose:
            print_load_confidence_intervals(
                partition_num, iteration, alpha_ini, node_index, confidence_interval
            )

    save_load_uncertainty_set(P_load_Uset, data_dir, num_nodes, alpha_ini)

    return P_load_Uset