from pathlib import Path
from cl_igdt import pre_processing_system_data as psd
from cl_igdt import processing_DERs_data as pder
from cl_igdt import CI_Demand as Demand
from cl_igdt import CI_PV as PV
from cl_igdt import economic_dispatch_IGDT as ed
import numpy as np


def load_input_data(data_dir, T):
    filename1 = data_dir / "IEEE69.xlsx"
    num_nodes = 69
    system_data = psd.pre_processing_system_data(filename1, num_nodes, T)

    filename2 = data_dir / "Data_69.xlsx"
    ders_data = pder.processing_DERs_data(filename2, T)

    return system_data, ders_data, num_nodes


def load_uncertainty_sets(data_dir, iteration, partition_num, T, α_ini, num_pv):
    try:
        with np.load(data_dir / f"Bus69_P_load_Uset_{α_ini}.npz") as data:
            P_load_Uset = {float(key): data[key] for key in data}
    except:
        P_load_Uset = Demand.CI_Demand(iteration, partition_num, T, α_ini)

    try:
        with np.load(data_dir / f"Bus69_PV_Uset_{α_ini}.npz") as data:
            PV_Uset = {float(key): data[key] for key in data}
    except:
        PV_Uset = PV.CI_PV(iteration, partition_num, T, α_ini, num_pv)

    return P_load_Uset, PV_Uset


def solve_one_iteration(
        system_data, ders_data, num_nodes, T,
        P_load_Uset, PV_Uset,
        iteration, partition_num, α_ini, accuracy
):
    α, _, cost_1S = ed.economic_dispatch_IGDT(
        system_data, ders_data, num_nodes, T,
        P_load_Uset, PV_Uset,
        iteration, partition_num, α_ini
    )

    temp = α_ini
    α_ini = round(np.nonzero(α)[0][0] * 10 ** (-iteration) + temp, accuracy)

    return α_ini, cost_1S


def run_ieee69():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    T = 24
    system_data, ders_data, num_nodes = load_input_data(data_dir, T)

    accuracy = 1
    partition_num = 10
    iteration = 1
    α_ini = 0

    while iteration <= accuracy:
        P_load_Uset, PV_Uset = load_uncertainty_sets(
            data_dir, iteration, partition_num, T, α_ini, len(ders_data[7])
        )

        α_ini, _ = solve_one_iteration(
            system_data, ders_data, num_nodes, T,
            P_load_Uset, PV_Uset,
            iteration, partition_num, α_ini, accuracy
        )

        print(f"The suboptimal confidence level of iteration {iteration} is：{α_ini}")

        iteration = iteration + 1


if __name__ == "__main__":
    run_ieee69()