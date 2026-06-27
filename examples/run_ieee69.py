from pathlib import Path
from cl_igdt import pre_processing_system_data as psd
from cl_igdt import processing_DERs_data as pder
from cl_igdt import CI_Demand as Demand
from cl_igdt import CI_PV as PV
from cl_igdt import economic_dispatch_IGDT as ed
import numpy as np
import time

if __name__=="__main__":
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

# ---------  Pre-processing System Data --------------
    filename1 = data_dir / "IEEE69.xlsx"
    num_nodes = 69
    T = 24
    System_Data = psd.pre_processing_system_data(filename1, num_nodes, T)

# ---------  Processing DERs Data --------------
    filename2 = data_dir / "Data_69.xlsx"
    DERs_Data = pder.processing_DERs_data(filename2, T)

# ---------  Construct and Solve the Model--------------
    accuracy = 1
    partition_num = 10
    iter = 1
    α_ini = 0
    total_time = 0

    while iter <= accuracy:
        start_time = time.time()

        #Load Demand uncertainty set
        try:
            with np.load(data_dir / f'Bus69_P_load_Uset_{α_ini}.npz') as data:
                P_load_Uset = {float(key): data[key] for key in data}
        except:
            P_load_Uset = Demand.CI_Demand(iter, partition_num, T, α_ini)

        # Load PV uncertainty set
        try:
            with np.load(data_dir / f'Bus69_PV_Uset_{α_ini}.npz') as data:
                PV_Uset = {float(key): data[key] for key in data}
        except:
            PV_Uset = PV.CI_PV(iter, partition_num, T, α_ini, len(DERs_Data[7]))

        # Construct the optimization model
        α, _, cost_1S = ed.economic_dispatch_IGDT(System_Data, DERs_Data, num_nodes, T, P_load_Uset, PV_Uset, iter, partition_num, α_ini)

        # Retrieve the results
        temp = α_ini
        α_ini = round(np.nonzero(α)[0][0] * 10**(-iter) + temp, accuracy)
        print(f'The suboptimal confidence level of iteration {iter} is：{α_ini}')

        end_time = time.time()  # 记录当前时间（循环结束）
        elapsed_time = end_time - start_time  # 计算本次循环花费的时间
        total_time += elapsed_time  # 累计总时间
        print(f'Iteration {iter} took {elapsed_time:.2f} seconds')
        print(f'Cumulative time after iteration {iter}: {total_time:.2f} seconds\n')  # 显示累计时间

        iter = iter + 1