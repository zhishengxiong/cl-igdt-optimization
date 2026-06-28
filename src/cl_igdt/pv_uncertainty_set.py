from collections import Counter

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import beta


PV_DATA_SHEET = "PVData"

IDM_CONFIDENCE_LEVEL = 0.95
IDM_EQUIVALENT_SAMPLE_SIZE = 1

PV_LOWER_EXTENSION_RATIO = 0.1
INTERPOLATION_KIND = "linear"
ZERO_VALUE = 0
FULL_PROBABILITY = 1


def read_historical_pv_data(data_dir, num_nodes):
    pv_data_file = data_dir / f"historical_data_PV_{num_nodes}.xlsx"

    if not pv_data_file.exists():
        raise FileNotFoundError(f"Historical PV data file not found: {pv_data_file}")

    available_sheets = pd.ExcelFile(pv_data_file).sheet_names

    if PV_DATA_SHEET not in available_sheets:
        raise ValueError(f"Missing sheet '{PV_DATA_SHEET}' in historical PV data file: {pv_data_file}")

    raw_pv_data = pd.read_excel(pv_data_file, sheet_name=PV_DATA_SHEET)

    return raw_pv_data


def build_idm_parameters():
    gamma = IDM_CONFIDENCE_LEVEL
    confidence_lower = (1 - gamma) / 2
    confidence_upper = (1 + gamma) / 2
    s = IDM_EQUIVALENT_SAMPLE_SIZE

    return confidence_lower, confidence_upper, s


def initialize_pv_uncertainty_set(partition_num, iteration, alpha_ini, T):
    pv_uset = {
        round(10 ** (-iteration) * partition + alpha_ini, iteration): np.zeros(T)
        for partition in range(partition_num)
    }

    return pv_uset


def cumulative_sum(number):
    result = []
    current_sum = 0

    for element in number:
        current_sum += element
        result.append(current_sum)

    return result


def extract_pv_samples(raw_pv_data):
    PV = {}
    PV_k = {}

    for _, row in raw_pv_data.iterrows():
        PV[row.iloc[0]] = [round(num, 1) for num in sorted(row.iloc[1:].tolist())]
        PV_k[row.iloc[0]] = sorted(set(PV[row.iloc[0]]))

    if "t0" not in PV:
        raise ValueError(f"Missing t0 row in sheet: {PV_DATA_SHEET}")

    value_counts = {key: Counter(values) for key, values in PV.items()}
    sample_counts = {key: list(element.values()) for key, element in value_counts.items()}

    n_k = {key: cumulative_sum(values) for key, values in sample_counts.items()}
    n = len(PV["t0"])

    return PV, PV_k, n_k, n


def build_idm_interpolated_cdf(PV_k, n_k, n, confidence_lower, confidence_upper, s):
    icdf_lower = {}
    icdf_upper = {}

    final_PV_k = {}
    final_icdf_lower = {}
    final_icdf_upper = {}

    for t, n_k_values in n_k.items():
        if PV_k[t][0] > ZERO_VALUE:
            PV_min = (PV_k[t][-1] - PV_k[t][0]) * PV_LOWER_EXTENSION_RATIO
            PV_min_new = round(PV_k[t][0] - PV_min, 1)
            PV_k[t].insert(0, max(ZERO_VALUE, PV_min_new))

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

        elif PV_k[t][-1] > ZERO_VALUE:
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

        if len(PV_k[t]) >= 2:
            f_inter_lower = interp1d(PV_k[t], icdf_lower[t], kind=INTERPOLATION_KIND)
            f_inter_upper = interp1d(PV_k[t], icdf_upper[t], kind=INTERPOLATION_KIND)

            x_z = [
                (PV_k[t][i] + PV_k[t][i + 1]) * 0.5
                for i in range(len(PV_k[t]) - 1)
            ]
            x_z = [round(num, 2) for num in x_z]

            final_PV_k[t] = sorted(PV_k[t] + x_z)
            final_icdf_lower[t] = f_inter_lower(final_PV_k[t])
            final_icdf_upper[t] = f_inter_upper(final_PV_k[t])

        else:
            final_PV_k[t] = PV_k[t]
            final_icdf_lower[t] = icdf_lower[t]
            final_icdf_upper[t] = icdf_upper[t]

    return final_PV_k, final_icdf_lower, final_icdf_upper


def initialize_confidence_interval(confidence_level, confidence_interval, T):
    if confidence_level not in confidence_interval:
        confidence_interval[confidence_level] = {
            f"t{i}": (ZERO_VALUE, ZERO_VALUE) for i in range(T)
        }

    return confidence_interval


def find_shortest_pv_interval(values, cum_probs_lower, cum_probs_upper, confidence_level):
    prob_densities = np.diff(cum_probs_upper) / np.diff(values)
    prob_densities = np.insert(prob_densities, 0, cum_probs_upper[0])

    max_density_idx = np.argmax(prob_densities)

    shortest_interval = (np.min(values), np.max(values))
    min_width = np.max(values) - np.min(values)

    if confidence_level == ZERO_VALUE:
        shortest_interval = (
            values[max_density_idx],
            values[max_density_idx],
        )
    else:
        if max_density_idx != ZERO_VALUE:
            for i in range(len(values) - 1):
                for j in range(i + 1, len(values)):
                    if cum_probs_lower[j] - cum_probs_lower[i] >= confidence_level:
                        width = values[j] - values[i]
                        if width < min_width:
                            min_width = width
                            shortest_interval = (values[i], values[j])
                        break
        else:
            for j in range(len(values)):
                if cum_probs_upper[j] >= confidence_level:
                    shortest_interval = (ZERO_VALUE, values[j])
                    break

    return shortest_interval


def build_shortest_pv_confidence_intervals(
        final_PV_k, final_icdf_lower, final_icdf_upper,
        partition_num, iteration, alpha_ini,
        pv_uset, T
):
    confidence_interval = {}

    for t, _ in final_PV_k.items():
        values = np.array(final_PV_k[t])
        cum_probs_lower = np.array(final_icdf_lower[t])
        cum_probs_upper = np.array(final_icdf_upper[t])

        for partition in range(partition_num):
            confidence_level = round(10 ** (-iteration) * partition + alpha_ini, iteration)

            confidence_interval = initialize_confidence_interval(
                confidence_level, confidence_interval, T
            )

            shortest_interval = find_shortest_pv_interval(
                values, cum_probs_lower, cum_probs_upper, confidence_level
            )

            pv_uset[confidence_level][int(t[1:])] = round(shortest_interval[0], 2)
            confidence_interval[confidence_level][t] = shortest_interval

    return pv_uset, confidence_interval


def tile_pv_uncertainty_set(pv_uset, partition_num, iteration, alpha_ini, num_pv):
    for partition in range(partition_num):
        confidence_level = round(10 ** (-iteration) * partition + alpha_ini, iteration)
        values = pv_uset[confidence_level]
        pv_uset[confidence_level] = np.tile(values, (num_pv, 1))

    return pv_uset


def print_pv_confidence_intervals(partition_num, iteration, alpha_ini, confidence_interval):
    for partition in range(partition_num):
        confidence_level = round(10 ** (-iteration) * partition + alpha_ini, iteration)
        print(
            f"The shortest CI of PV of confidence level {confidence_level} is:",
            confidence_interval[confidence_level],
        )


def save_pv_uncertainty_set(pv_uset, data_dir, num_nodes, alpha_ini):
    output_file = data_dir / f"Bus{num_nodes}_PV_Uset_{alpha_ini}.npz"

    np.savez(
        output_file,
        **{str(key): value for key, value in pv_uset.items()}
    )


def build_pv_uncertainty_set(
        data_dir, num_nodes, iteration, partition_num, T, alpha_ini, num_pv,
        verbose=False
):
    raw_pv_data = read_historical_pv_data(data_dir, num_nodes)

    confidence_lower, confidence_upper, s = build_idm_parameters()

    pv_uset = initialize_pv_uncertainty_set(
        partition_num, iteration, alpha_ini, T
    )

    PV, PV_k, n_k, n = extract_pv_samples(raw_pv_data)

    final_PV_k, final_icdf_lower, final_icdf_upper = build_idm_interpolated_cdf(
        PV_k, n_k, n,
        confidence_lower, confidence_upper, s
    )

    pv_uset, confidence_interval = build_shortest_pv_confidence_intervals(
        final_PV_k, final_icdf_lower, final_icdf_upper,
        partition_num, iteration, alpha_ini,
        pv_uset, T
    )

    pv_uset = tile_pv_uncertainty_set(
        pv_uset, partition_num, iteration, alpha_ini, num_pv
    )

    if verbose:
        print_pv_confidence_intervals(
            partition_num, iteration, alpha_ini, confidence_interval
        )

    save_pv_uncertainty_set(pv_uset, data_dir, num_nodes, alpha_ini)

    return pv_uset