from pathlib import Path
from scipy.stats import beta
import pandas as pd
from collections import Counter
import numpy as np
from scipy.interpolate import interp1d


def read_historical_pv_data():
    project_root = Path(__file__).resolve().parents[2]
    filename = project_root / "data" / "historcial data_PV.xlsx"

    raw_pv_generation = pd.read_excel(filename, sheet_name="PVData")

    return raw_pv_generation


def build_idm_parameters():
    gamma = 0.95
    confidence_lower = (1 - gamma) / 2
    confidence_upper = (1 + gamma) / 2
    s = 1

    return confidence_lower, confidence_upper, s


def initialize_pv_uncertainty_set(partition_num, iter, α_ini, T):
    pv_uset = {
        round(10 ** (-iter) * partition + α_ini, iter): np.zeros(T)
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


def extract_pv_samples(raw_pv_generation):
    PV = {}
    PV_k = {}

    for _, row in raw_pv_generation.iterrows():
        PV[row.iloc[0]] = [round(num, 1) for num in sorted(row.iloc[1:].tolist())]
        PV_k[row.iloc[0]] = sorted(set(PV[row.iloc[0]]))

    temp = {key: Counter(values) for key, values in PV.items()}
    number = {key: list(element.values()) for key, element in temp.items()}

    n_k = {key: cumulative_sum(values) for key, values in number.items()}
    n = len(PV["t0"])

    return PV, PV_k, n_k, n


def build_idm_interpolated_cdf(PV_k, n_k, n, confidence_lower, confidence_upper, s):
    icdf_lower = {}
    icdf_upper = {}

    final_PV_k = {}
    final_icdf_lower = {}
    final_icdf_upper = {}

    for t, n_k_values in n_k.items():
        if PV_k[t][0] > 0:
            PV_min = (PV_k[t][-1] - PV_k[t][0]) * 0.1
            PV_min_new = round(PV_k[t][0] - PV_min, 1)
            PV_k[t].insert(0, max(0, PV_min_new))

            icdf_lower[t] = [
                beta.ppf(confidence_lower, i, s + n - i)
                for i in n_k_values
            ]
            icdf_lower[t].insert(0, 0)

            n_k_values = n_k_values[:-1]
            n_k_values.insert(0, 0)
            icdf_upper[t] = [
                beta.ppf(confidence_upper, s + i, n - i)
                for i in n_k_values
            ]
            icdf_upper[t].append(1)

        elif PV_k[t][-1] > 0:
            icdf_lower[t] = [
                beta.ppf(confidence_lower, i, s + n - i)
                for i in n_k_values
            ]

            n_k_values = n_k_values[:-1]
            icdf_upper[t] = [
                beta.ppf(confidence_upper, s + i, n - i)
                for i in n_k_values
            ]
            icdf_upper[t].append(1)

        else:
            icdf_lower[t] = [1]
            icdf_upper[t] = [1]

        if len(PV_k[t]) >= 2:
            f_inter_lower = interp1d(PV_k[t], icdf_lower[t], kind="linear")
            f_inter_upper = interp1d(PV_k[t], icdf_upper[t], kind="linear")

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


def build_shortest_pv_confidence_intervals(
        final_PV_k, final_icdf_lower, final_icdf_upper,
        partition_num, iter, α_ini,
        pv_uset
):
    confidence_interval = {}

    for t, _ in final_PV_k.items():
        values = np.array(final_PV_k[t])
        cum_probs_lower = np.array(final_icdf_lower[t])
        cum_probs_upper = np.array(final_icdf_upper[t])

        prob_densities = np.diff(cum_probs_upper) / np.diff(values)
        prob_densities = np.insert(prob_densities, 0, final_icdf_upper[t][0])

        max_density_idx = np.argmax(prob_densities)

        for partition in range(partition_num):
            confidence_level = round(10 ** (-iter) * partition + α_ini, iter)

            if confidence_level not in confidence_interval:
                confidence_interval[confidence_level] = {
                    f"t{i}": (0, 0) for i in range(24)
                }

            shortest_interval = (np.min(values), np.max(values))
            min_width = np.max(values) - np.min(values)

            if confidence_level == 0:
                shortest_interval = (
                    values[max_density_idx],
                    values[max_density_idx]
                )
            else:
                if max_density_idx != 0:
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
                            shortest_interval = (0, values[j])
                            break

            pv_uset[confidence_level][int(t[1:])] = round(shortest_interval[0], 2)
            confidence_interval[confidence_level][t] = shortest_interval

    return pv_uset, confidence_interval


def tile_pv_uncertainty_set(pv_uset, partition_num, iter, α_ini, num_PV):
    for partition in range(partition_num):
        confidence_level = round(10 ** (-iter) * partition + α_ini, iter)
        values = pv_uset[confidence_level]
        pv_uset[confidence_level] = np.tile(values, (num_PV, 1))

    return pv_uset


def print_pv_confidence_intervals(partition_num, iter, α_ini, confidence_interval):
    for partition in range(partition_num):
        confidence_level = round(10 ** (-iter) * partition + α_ini, iter)
        print(
            f"The shortest CI of PV of confidence level {confidence_level} is:",
            confidence_interval[confidence_level]
        )


def save_pv_uncertainty_set(pv_uset, α_ini):
    np.savez(
        f"Bus69_PV_Uset_{α_ini}.npz",
        **{str(key): value for key, value in pv_uset.items()}
    )


def CI_PV(iter, partition_num, T, α_ini, num_PV):
    raw_pv_generation = read_historical_pv_data()

    confidence_lower, confidence_upper, s = build_idm_parameters()

    pv_uset = initialize_pv_uncertainty_set(
        partition_num, iter, α_ini, T
    )

    PV, PV_k, n_k, n = extract_pv_samples(raw_pv_generation)

    final_PV_k, final_icdf_lower, final_icdf_upper = build_idm_interpolated_cdf(
        PV_k, n_k, n,
        confidence_lower, confidence_upper, s
    )

    pv_uset, confidence_interval = build_shortest_pv_confidence_intervals(
        final_PV_k, final_icdf_lower, final_icdf_upper,
        partition_num, iter, α_ini,
        pv_uset
    )

    pv_uset = tile_pv_uncertainty_set(
        pv_uset, partition_num, iter, α_ini, num_PV
    )

    print_pv_confidence_intervals(
        partition_num, iter, α_ini, confidence_interval
    )

    save_pv_uncertainty_set(pv_uset, α_ini)

    return pv_uset