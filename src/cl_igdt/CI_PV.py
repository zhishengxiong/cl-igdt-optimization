#####This is the function to fit the CDF of PV using IDM
#  and obtain the shortest confidence interval#####
from scipy.stats import beta
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import numpy as np
from scipy.interpolate import interp1d

def CI_PV(iter, partition_num, T, α_ini, num_PV):
##----------------Initialization----------------
    PV_Uset = {round(10 ** (-iter) * partition + α_ini, iter): np.zeros(T) for partition in range(partition_num)}

    gamma = 0.95
    confidence_lower = (1 - gamma) / 2
    confidence_upper = (1 + gamma) / 2
    s = 1

##----------------Construct the ambiguity set----------------
    ##extract data from the excel document
    PVGeneration = pd.read_excel('historcial data_PV.xlsx', sheet_name='PVData')

    PV = {}
    PV_k = {}
    for _, row in PVGeneration.iterrows():
        PV[row.iloc[0]] = [round(num,1) for num in sorted(row.iloc[1:].tolist())]
        PV_k[row.iloc[0]]= sorted(set(PV[row.iloc[0]]))

    ##obtain the unique value and their respective frequency of occurrences
    temp = {key: Counter(values) for key, values in PV.items()}
    number = {key: list(element.values()) for key, element in temp.items()}

    def cumulative_sum(number):
        result = []
        current_sum = 0
        for element in number:
            current_sum += element
            result.append(current_sum)
        return result

    n_k = {key: cumulative_sum(values) for key, values in number.items()}
    n = len(PV['t0'])

    ## Imprecise Dirichlet model (IDM)
    icdf_lower = {} #store the values of IDM
    icdf_upper = {}

    final_PV_k = {} #store the values of IDM after interpolation
    final_icdf_lower = {}
    final_icdf_upper = {}

       #Calculate the lower and upper bound of CDF
        #determine if there are some (or all)zeros in the data
    for t, n_k_values in n_k.items():
        if PV_k[t][0] > 0:   #all the output is positive
                # correct the actual interval of PV
            PV_min = (PV_k[t][-1] - PV_k[t][0]) * 0.1
            PV_min_new = round(PV_k[t][0] - PV_min, 1)
            PV_k[t].insert(0, max(0, PV_min_new))

                # lower bound
            icdf_lower[t] = [beta.ppf(confidence_lower, i, s + n - i) for i in n_k_values]
            icdf_lower[t].insert(0, 0)

                #upper bound
            n_k_values = n_k_values[:-1]
            n_k_values.insert(0, 0)
            icdf_upper[t] = [beta.ppf(confidence_upper, s + i, n - i) for i in n_k_values]
            icdf_upper[t].append(1)

        elif PV_k[t][-1] > 0: #a part of the output is zero
                    # lower bound
            icdf_lower[t] = [beta.ppf(confidence_lower, i, s + n - i) for i in n_k_values]

                    # upper bound
            n_k_values = n_k_values[:-1]
            icdf_upper[t] = [beta.ppf(confidence_upper, s + i, n - i) for i in n_k_values]
            icdf_upper[t].append(1)

        else: #all the output are zeros
            icdf_lower[t] = [1]
            icdf_upper[t] = [1]

        ##Interpolation methods
        if len(PV_k[t]) >= 2:
            f_inter_lower = interp1d(PV_k[t], icdf_lower[t], kind='linear')
            f_inter_upper = interp1d(PV_k[t], icdf_upper[t], kind='linear')

            x_z = [(PV_k[t][i] + PV_k[t][i+1]) * 0.5 for i in range(len(PV_k[t]) - 1)]
            x_z = [round(num,2) for num in x_z]
            final_PV_k[t] = sorted(PV_k[t] + x_z)

            final_icdf_lower[t] = f_inter_lower(final_PV_k[t])
            final_icdf_upper[t] = f_inter_upper(final_PV_k[t])
        else:
            final_PV_k[t] = PV_k[t]
            final_icdf_lower[t] = icdf_lower[t]
            final_icdf_upper[t] = icdf_upper[t]

##----------------calculate the shortest confidence interval----------------
    confidence_interval = {}

    for t, _ in final_PV_k.items():
        values = np.array(final_PV_k[t])
        cum_probs_lower = np.array(final_icdf_lower[t])
        cum_probs_upper = np.array(final_icdf_upper[t])

        ## Calculate the probality density of each point
        prob_densities = np.diff(cum_probs_upper) / np.diff(values)
        prob_densities = np.insert(prob_densities, 0, final_icdf_upper[t][0])

        ## Search the point with the largest probability density
        max_density_idx = np.argmax(prob_densities)

        for partition in range(partition_num):
            confidence_level = round(10 ** (-iter) * partition + α_ini, iter)
            if confidence_level not in confidence_interval:
                confidence_interval[confidence_level] = {f't{i}': (0, 0) for i in range(24)}

            ## Initialize the shortest CI (the whole interval)
            shortest_interval = (np.min(values), np.max(values))
            min_width = np.max(values) - np.min(values)

            ## Search the shortest CI
            if confidence_level == 0:
                shortest_interval = (values[max_density_idx], values[max_density_idx])
            else:
                if max_density_idx != 0:
                    for i in range(len(values)-1):
                        for j in range(i+1, len(values)):
                              # Check whether the value of the first point is positive
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

            PV_Uset[confidence_level][int(t[1:])] = round(shortest_interval[0], 2)
            confidence_interval[confidence_level][t] = shortest_interval

    for partition in range(partition_num):
        confidence_level = round(10 ** (-iter) * partition + α_ini, iter)
        values = PV_Uset[confidence_level]
        PV_Uset[confidence_level] = np.tile(values, (num_PV, 1))
        print(f"The shortest CI of PV of confidence level {confidence_level} is:", confidence_interval[confidence_level])

    # save PV_Uset for future direct loading
    np.savez(f'Bus69_PV_Uset_{α_ini}.npz',
             **{str(key): value for key, value in PV_Uset.items()})

    return PV_Uset