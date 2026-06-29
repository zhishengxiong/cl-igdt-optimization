"""Formulate and solve the CL-IGDT economic dispatch model.

The model includes first-stage decisions, second-stage recourse decisions,
network constraints, uncertainty realization constraints, and the IGDT objective.
"""

from dataclasses import dataclass

import numpy as np
from gurobipy import GRB, Model, quicksum


@dataclass
class OptimalResult:
    theta_u: np.ndarray
    total_cost: float
    first_stage_cost: float
    G_p: np.ndarray
    ESS_pch: np.ndarray
    ESS_pdis: np.ndarray
    ESS_u: np.ndarray
    P_flow: np.ndarray


def unpack_system_data(system_data):
    A = system_data.A
    R_prime = system_data.R_prime * 0.001
    X_prime = system_data.X_prime * 0.001
    P_load = system_data.P_load

    return A, R_prime, X_prime, P_load


def unpack_ders_data(ders_data, T):
    G_node = ders_data.G_node

    G_pmax = np.tile(ders_data.G_pmax, (1, T))
    G_pmin = np.tile(ders_data.G_pmin, (1, T))
    G_qmax = np.tile(ders_data.G_qmax, (1, T))
    G_qmin = np.tile(ders_data.G_qmin, (1, T))

    G_cost = ders_data.G_cost
    electricity_price = ders_data.electricity_price

    PV_node = ders_data.PV_node
    PV = ders_data.PV

    G_up_limit = ders_data.G_up_limit
    G_dn_limit = ders_data.G_dn_limit

    ESS_node = ders_data.ESS_node
    ESS_pmax = ders_data.ESS_pmax
    ESS_capacity = ders_data.ESS_capacity
    ESS_Eini = ders_data.ESS_Eini
    ESS_eff = ders_data.ESS_eff

    return (
        G_node,
        G_pmax,
        G_pmin,
        G_qmax,
        G_qmin,
        G_cost,
        electricity_price,
        PV_node,
        PV,
        G_up_limit,
        G_dn_limit,
        ESS_node,
        ESS_pmax,
        ESS_capacity,
        ESS_Eini,
        ESS_eff,
    )


def build_opt_model_limits(num_nodes, network_config, optimization_config):
    num_non_slack_nodes = num_nodes - 1

    base_voltage_kv = network_config["base_voltage_kv"]
    voltage_upper_limit = optimization_config["voltage_upper_limit"]
    voltage_lower_limit = optimization_config["voltage_lower_limit"]
    P_flow_up = optimization_config["p_flow_upper_limit"]

    v0 = base_voltage_kv * base_voltage_kv

    v_up = voltage_upper_limit**2 * v0
    v_low = voltage_lower_limit**2 * v0

    P_flow_low = -P_flow_up

    return num_non_slack_nodes, v0, v_up, v_low, P_flow_up, P_flow_low


def create_opt_model():
    m = Model("CL_IGDT")
    m.setParam("OutputFlag", 0)

    return m


def create_generator_variables(m, T, G_node, G_pmin, G_pmax, G_qmin, G_qmax, optimization_config):
    generator_second_stage_adjustment_ratio = optimization_config[
        "generator_second_stage_adjustment_ratio"
    ]

    G_p = m.addMVar((len(G_node), T), lb=G_pmin, ub=G_pmax, vtype=GRB.CONTINUOUS, name="G_p")
    G_q = m.addMVar((len(G_node), T), lb=G_qmin, ub=G_qmax, vtype=GRB.CONTINUOUS, name="G_q")

    G_p_cor = m.addMVar(
        (len(G_node), T),
        lb=-G_pmax * generator_second_stage_adjustment_ratio,
        ub=G_pmax * generator_second_stage_adjustment_ratio,
        vtype=GRB.CONTINUOUS,
        name="G_p_cor",
    )
    G_p_cor_abs = m.addMVar(
        (len(G_node), T),
        lb=-GRB.INFINITY,
        ub=GRB.INFINITY,
        vtype=GRB.CONTINUOUS,
        name="G_p_cor_abs",
    )
    G_p_reg = m.addMVar(
        (len(G_node), T), lb=G_pmin, ub=G_pmax, vtype=GRB.CONTINUOUS, name="G_p_reg"
    )

    return G_p, G_q, G_p_cor, G_p_cor_abs, G_p_reg


def create_ess_variables(m, T, ESS_node, ESS_pmax, ESS_capacity, optimization_config):
    ess_min_soc_ratio = optimization_config["ess_min_soc_ratio"]

    ESS_pch = m.addMVar((len(ESS_node), T), lb=0, vtype=GRB.CONTINUOUS, name="ESS_ch")
    ESS_pdis = m.addMVar((len(ESS_node), T), lb=0, vtype=GRB.CONTINUOUS, name="ESS_dis")
    ESS_E = m.addMVar(
        (len(ESS_node), T),
        lb=ESS_capacity * ess_min_soc_ratio,
        ub=ESS_capacity,
        vtype=GRB.CONTINUOUS,
        name="ESS_E",
    )
    ESS_u = m.addMVar((len(ESS_node), T), vtype=GRB.BINARY, name="ESS_u")

    return ESS_pch, ESS_pdis, ESS_E, ESS_u


def create_network_variables(m, num_non_slack_nodes, T, v_low, v_up, P_flow_low, P_flow_up):
    P_flow = m.addMVar((num_non_slack_nodes, T), lb=P_flow_low, ub=P_flow_up, name="P_flow")
    P_flow_2S = m.addMVar((num_non_slack_nodes, T), lb=P_flow_low, ub=P_flow_up, name="P_flow_2S")
    Q_flow = m.addMVar((num_non_slack_nodes, T), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Q_flow")
    v = m.addMVar((num_non_slack_nodes, T), lb=v_low, ub=v_up, name="v")

    P_net = m.addMVar((num_non_slack_nodes, T), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="P_net")
    P_net_2S = m.addMVar(
        (num_non_slack_nodes, T), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="P_net_2S"
    )
    Q_net = m.addMVar((num_non_slack_nodes, T), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Q_net")

    return P_flow, P_flow_2S, Q_flow, v, P_net, P_net_2S, Q_net


def create_uncertain_realization_variables(m, num_non_slack_nodes, T, PV_node):
    P_load_real = m.addMVar(
        (num_non_slack_nodes, T),
        lb=-GRB.INFINITY,
        ub=GRB.INFINITY,
        vtype=GRB.CONTINUOUS,
        name="P_load_real",
    )
    Q_load = m.addMVar(
        (num_non_slack_nodes, T),
        lb=-GRB.INFINITY,
        ub=GRB.INFINITY,
        vtype=GRB.CONTINUOUS,
        name="Q_load",
    )
    PV_real = m.addMVar(
        (len(PV_node), T),
        lb=-GRB.INFINITY,
        ub=GRB.INFINITY,
        vtype=GRB.CONTINUOUS,
        name="PV_real",
    )

    return P_load_real, Q_load, PV_real


def create_power_exchange_auxiliary_variables(m, T):
    flow_pos_auxi = m.addMVar(T, lb=0, vtype=GRB.CONTINUOUS, name="flow_pos_auxi")
    flow_neg_auxi = m.addMVar(T, lb=0, vtype=GRB.CONTINUOUS, name="flow_neg_auxi")
    flow_u = m.addMVar(T, vtype=GRB.BINARY, name="flow_u")

    return flow_pos_auxi, flow_neg_auxi, flow_u


def create_igdt_obj_variables(m, partition_num):
    theta = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="theta")
    theta_s = m.addMVar(partition_num, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="theta_s")
    theta_u = m.addMVar(partition_num, vtype=GRB.BINARY, name="theta_s_state")

    return theta, theta_s, theta_u


def add_first_stage_generator_constraints(m, T, G_p, G_up_limit, G_dn_limit):
    for t in range(1, T):
        m.addConstr(G_p[:, t] - G_p[:, t - 1] <= G_up_limit, name=f"G_p_up_rule_{t}")
        m.addConstr(G_p[:, t - 1] - G_p[:, t] <= G_dn_limit, name=f"G_p_dn_rule_{t}")


def add_first_stage_ess_constraints(
    m, T, ESS_node, ESS_E, ESS_Eini, ESS_eff, ESS_pch, ESS_pdis, ESS_u, ESS_pmax
):
    for t in range(T):
        for i in range(len(ESS_node)):
            if t == 0:
                m.addConstr(
                    ESS_E[i, t]
                    == ESS_Eini[i] + ESS_eff[i] * ESS_pch[i, t] - ESS_pdis[i, t] / ESS_eff[i],
                    name=f"ESS_energy_rule_{i}_{t}",
                )
            else:
                m.addConstr(
                    ESS_E[i, t]
                    == ESS_E[i, t - 1] + ESS_eff[i] * ESS_pch[i, t] - ESS_pdis[i, t] / ESS_eff[i],
                    name=f"ESS_energy_rule_{i}_{t}",
                )

            m.addConstr(ESS_pch[i, t] <= ESS_u[i, t] * ESS_pmax[i], name=f"ESS_ch_limit_{i}_{t}")
            m.addConstr(
                ESS_pdis[i, t] <= (1 - ESS_u[i, t]) * ESS_pmax[i],
                name=f"ESS_dis_limit_{i}_{t}",
            )


def add_first_stage_active_power_balance_constraints(
    m,
    num_non_slack_nodes,
    T,
    G_node,
    PV_node,
    ESS_node,
    P_load,
    PV,
    G_p,
    ESS_pdis,
    ESS_pch,
    P_net,
    P_flow,
    A,
):
    for n in range(num_non_slack_nodes):
        if n + 1 in G_node:
            for t in range(T):
                i = G_node.index(n + 1)
                m.addConstr(P_net[n, t] == P_load[n, t] + G_p[i, t], name=f"P_net_rule_{n}_{t}")

        elif n + 1 in PV_node:
            for t in range(T):
                i = PV_node.index(n + 1)
                m.addConstr(P_net[n, t] == P_load[n, t] + PV[i, t], name=f"P_net_rule_{n}_{t}")

        elif n + 1 in ESS_node:
            for t in range(T):
                i = ESS_node.index(n + 1)
                m.addConstr(
                    P_net[n, t] == P_load[n, t] + ESS_pdis[i, t] - ESS_pch[i, t],
                    name=f"P_net_rule_{n}_{t}",
                )

        else:
            for t in range(T):
                m.addConstr(P_net[n, t] == P_load[n, t], name=f"P_net_rule_{n}_{t}")

    for t in range(T):
        m.addConstr(A.T @ P_flow[:, t] == P_net[:, t], name=f"P_Ban_{t}")


def add_second_stage_generator_constraints(
    m, T, G_p, G_p_cor, G_p_cor_abs, G_p_reg, G_up_limit, G_dn_limit
):
    m.addConstr(G_p_reg == G_p + G_p_cor, name="G_reg_rule")
    m.addConstr(G_p_cor_abs >= G_p_cor, name="G_abs_pos_rule")
    m.addConstr(G_p_cor_abs >= -G_p_cor, name="G_abs_neg_rule")

    for t in range(1, T):
        m.addConstr(G_p_reg[:, t] - G_p_reg[:, t - 1] <= G_up_limit, name=f"G_reg_up_rule_{t}")
        m.addConstr(G_p_reg[:, t - 1] - G_p_reg[:, t] <= G_dn_limit, name=f"G_reg_dn_rule_{t}")


def add_second_stage_net_loads_constraints(
    m,
    num_non_slack_nodes,
    T,
    G_node,
    PV_node,
    ESS_node,
    P_load_real,
    Q_load,
    PV_real,
    G_p_reg,
    G_q,
    ESS_pdis,
    ESS_pch,
    P_net_2S,
    Q_net,
):
    for n in range(num_non_slack_nodes):
        if n + 1 in G_node:
            for t in range(T):
                i = G_node.index(n + 1)
                m.addConstr(
                    P_net_2S[n, t] == P_load_real[n, t] + G_p_reg[i, t],
                    name=f"P_net_2S_rule_{n}_{t}",
                )
                m.addConstr(Q_net[n, t] == Q_load[n, t] + G_q[i, t], name=f"Q_net_rule_{n}_{t}")

        elif n + 1 in PV_node:
            for t in range(T):
                i = PV_node.index(n + 1)
                m.addConstr(
                    P_net_2S[n, t] == P_load_real[n, t] + PV_real[i, t],
                    name=f"P_net_2S_rule_{n}_{t}",
                )
                m.addConstr(Q_net[n, t] == Q_load[n, t], name=f"Q_net_rule_{n}_{t}")

        elif n + 1 in ESS_node:
            for t in range(T):
                i = ESS_node.index(n + 1)
                m.addConstr(
                    P_net_2S[n, t] == P_load_real[n, t] + ESS_pdis[i, t] - ESS_pch[i, t],
                    name=f"P_net_2S_rule_{n}_{t}",
                )
                m.addConstr(Q_net[n, t] == Q_load[n, t], name=f"Q_net_rule_{n}_{t}")

        else:
            for t in range(T):
                m.addConstr(P_net_2S[n, t] == P_load_real[n, t], name=f"P_net_2S_rule_{n}_{t}")
                m.addConstr(Q_net[n, t] == Q_load[n, t], name=f"Q_net_rule_{n}_{t}")


def add_second_stage_network_constraints(
    m, T, A, R_prime, X_prime, v0, P_flow_2S, Q_flow, P_net_2S, Q_net, v
):
    for t in range(T):
        m.addConstr(A.T @ P_flow_2S[:, t] == P_net_2S[:, t], name=f"P_Ban_2S_{t}")
        m.addConstr(A.T @ Q_flow[:, t] == Q_net[:, t], name=f"Q_Ban_{t}")
        m.addConstr(
            v[:, t] == v0 + (R_prime @ P_net_2S[:, t] + X_prime @ Q_net[:, t]),
            name=f"v_rule_{t}",
        )


def add_igdt_partition_constraints(m, partition_num, iteration, alpha_ini, theta, theta_s, theta_u):
    for i in range(partition_num):
        m.addConstr(
            theta_s[i] >= (alpha_ini + 10 ** (-iteration) * i) * theta_u[i],
            name=f"theta_s_limit1_{i}",
        )
        m.addConstr(
            theta_s[i] <= (alpha_ini + 10 ** (-iteration) * (i + 1)) * theta_u[i],
            name=f"theta_s_limit2_{i}",
        )

    m.addConstr(quicksum(theta_u[i] for i in range(partition_num)) == 1, name="obj_u_rule")
    m.addConstr(quicksum(theta_s[i] for i in range(partition_num)) == theta, name="theta_rule")


def add_uncertainty_realization_constraints(
    m,
    partition_num,
    iteration,
    alpha_ini,
    P_load_Uset,
    PV_Uset,
    P_load_real,
    Q_load,
    PV_real,
    theta_u,
    optimization_config,
):
    power_factor = optimization_config["power_factor"]

    m.addConstr(
        P_load_real
        == quicksum(
            P_load_Uset[round(10 ** (-iteration) * i + alpha_ini, iteration)] * theta_u[i]
            for i in range(partition_num)
        ),
        name="Load_Uset_rule",
    )

    m.addConstr(Q_load == P_load_real * power_factor, name="Reactive_Load_rule")

    m.addConstr(
        PV_real
        == quicksum(
            PV_Uset[round(10 ** (-iteration) * i + alpha_ini, iteration)] * theta_u[i]
            for i in range(partition_num)
        ),
        name="PV_Uset_rule",
    )


def build_total_cost(
    T,
    G_p,
    G_p_cor_abs,
    P_flow,
    flow_pos_auxi,
    flow_neg_auxi,
    G_cost,
    electricity_price,
    optimization_config,
):
    second_stage_generation_price_ratio = optimization_config["second_stage_generation_price_ratio"]
    second_stage_import_price_ratio = optimization_config["second_stage_import_price_ratio"]
    second_stage_export_price_ratio = optimization_config["second_stage_export_price_ratio"]

    total_cost = (
        quicksum(G_p[:, t] for t in range(T)) @ G_cost
        + quicksum(G_p_cor_abs[:, t] for t in range(T))
        @ G_cost
        * second_stage_generation_price_ratio
        + quicksum(P_flow[0, t] * electricity_price[t] for t in range(T))
        + quicksum(
            flow_pos_auxi[t] * electricity_price[t] * second_stage_import_price_ratio
            for t in range(T)
        )
        - quicksum(
            flow_neg_auxi[t] * electricity_price[t] * second_stage_export_price_ratio
            for t in range(T)
        )
    )

    return total_cost


def add_budget_constraint(m, total_cost, optimization_config):
    budget_base_cost = optimization_config["budget_base_cost"]
    budget_deviation_ratio = optimization_config["budget_deviation_ratio"]

    m.addConstr(total_cost <= budget_base_cost * budget_deviation_ratio, name="Budget_limit")


def add_power_exchange_auxiliary_constraints(
    m, T, P_flow, P_flow_2S, flow_pos_auxi, flow_neg_auxi, flow_u, P_flow_up
):
    for t in range(T):
        m.addConstr(
            flow_pos_auxi[t] - flow_neg_auxi[t] == P_flow_2S[0, t] - P_flow[0, t],
            name="Flow_diff_rule",
        )
        m.addConstr(flow_pos_auxi[t] <= flow_u[t] * P_flow_up * 2, name="Flow_abs_pos_rule")
        m.addConstr(
            flow_neg_auxi[t] <= (1 - flow_u[t]) * P_flow_up * 2,
            name="Flow_abs_neg_rule",
        )


def set_igdt_objective(m, theta):
    m.setObjective(theta, GRB.MAXIMIZE)


def solve_and_extract_results(
    m,
    T,
    G_p,
    G_p_cor_abs,
    ESS_pch,
    ESS_pdis,
    ESS_u,
    P_flow,
    flow_pos_auxi,
    flow_neg_auxi,
    G_cost,
    electricity_price,
    theta_u,
    optimization_config,
):
    second_stage_generation_price_ratio = optimization_config["second_stage_generation_price_ratio"]
    second_stage_import_price_ratio = optimization_config["second_stage_import_price_ratio"]
    second_stage_export_price_ratio = optimization_config["second_stage_export_price_ratio"]

    m.optimize()

    if m.status != GRB.OPTIMAL:
        return None

    total_cost_spent = (
        quicksum(G_p.x[:, t] for t in range(T)) @ G_cost
        + quicksum(G_p_cor_abs.x[:, t] for t in range(T))
        @ G_cost
        * second_stage_generation_price_ratio
        + quicksum(P_flow.x[0, t] * electricity_price[t] for t in range(T))
        + quicksum(
            flow_pos_auxi.x[t] * electricity_price[t] * second_stage_import_price_ratio
            for t in range(T)
        )
        - quicksum(
            flow_neg_auxi.x[t] * electricity_price[t] * second_stage_export_price_ratio
            for t in range(T)
        )
    )

    first_stage_cost = quicksum(G_p.x[:, t] for t in range(T)) @ G_cost + quicksum(
        P_flow.x[0, t] * electricity_price[t] for t in range(T)
    )

    optimal_result = OptimalResult(
        theta_u=theta_u.x,
        total_cost=round(total_cost_spent.getValue(), 2),
        first_stage_cost=round(first_stage_cost.getValue(), 2),
        G_p=G_p.x,
        ESS_pch=ESS_pch.x,
        ESS_pdis=ESS_pdis.x,
        ESS_u=ESS_u.x,
        P_flow=P_flow.x,
    )

    return optimal_result


def solve_economic_dispatch_igdt(
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
):
    """Solve optimization model for the current alpha step."""

    # Parameters
    A, R_prime, X_prime, P_load = unpack_system_data(system_data)

    (
        G_node,
        G_pmax,
        G_pmin,
        G_qmax,
        G_qmin,
        G_cost,
        electricity_price,
        PV_node,
        PV,
        G_up_limit,
        G_dn_limit,
        ESS_node,
        ESS_pmax,
        ESS_capacity,
        ESS_Eini,
        ESS_eff,
    ) = unpack_ders_data(ders_data, T)

    num_non_slack_nodes, v0, v_up, v_low, P_flow_up, P_flow_low = build_opt_model_limits(
        num_nodes, network_config, optimization_config
    )

    # Decision variables
    m = create_opt_model()

    G_p, G_q, G_p_cor, G_p_cor_abs, G_p_reg = create_generator_variables(
        m, T, G_node, G_pmin, G_pmax, G_qmin, G_qmax, optimization_config
    )

    ESS_pch, ESS_pdis, ESS_E, ESS_u = create_ess_variables(
        m, T, ESS_node, ESS_pmax, ESS_capacity, optimization_config
    )

    P_flow, P_flow_2S, Q_flow, v, P_net, P_net_2S, Q_net = create_network_variables(
        m, num_non_slack_nodes, T, v_low, v_up, P_flow_low, P_flow_up
    )

    P_load_real, Q_load, PV_real = create_uncertain_realization_variables(
        m, num_non_slack_nodes, T, PV_node
    )

    flow_pos_auxi, flow_neg_auxi, flow_u = create_power_exchange_auxiliary_variables(m, T)

    theta, theta_s, theta_u = create_igdt_obj_variables(m, partition_num)

    # First-stage constraints
    add_first_stage_generator_constraints(m, T, G_p, G_up_limit, G_dn_limit)

    add_first_stage_ess_constraints(
        m, T, ESS_node, ESS_E, ESS_Eini, ESS_eff, ESS_pch, ESS_pdis, ESS_u, ESS_pmax
    )

    add_first_stage_active_power_balance_constraints(
        m,
        num_non_slack_nodes,
        T,
        G_node,
        PV_node,
        ESS_node,
        P_load,
        PV,
        G_p,
        ESS_pdis,
        ESS_pch,
        P_net,
        P_flow,
        A,
    )

    # Second-stage constraints
    add_second_stage_generator_constraints(
        m, T, G_p, G_p_cor, G_p_cor_abs, G_p_reg, G_up_limit, G_dn_limit
    )

    add_second_stage_net_loads_constraints(
        m,
        num_non_slack_nodes,
        T,
        G_node,
        PV_node,
        ESS_node,
        P_load_real,
        Q_load,
        PV_real,
        G_p_reg,
        G_q,
        ESS_pdis,
        ESS_pch,
        P_net_2S,
        Q_net,
    )

    add_second_stage_network_constraints(
        m, T, A, R_prime, X_prime, v0, P_flow_2S, Q_flow, P_net_2S, Q_net, v
    )

    # Uncertainty set constraints
    # Partition of objective function
    add_igdt_partition_constraints(m, partition_num, iteration, alpha_ini, theta, theta_s, theta_u)

    # Partition of uncertainty set
    add_uncertainty_realization_constraints(
        m,
        partition_num,
        iteration,
        alpha_ini,
        P_load_Uset,
        PV_Uset,
        P_load_real,
        Q_load,
        PV_real,
        theta_u,
        optimization_config,
    )

    # Budget constraint
    total_cost = build_total_cost(
        T,
        G_p,
        G_p_cor_abs,
        P_flow,
        flow_pos_auxi,
        flow_neg_auxi,
        G_cost,
        electricity_price,
        optimization_config,
    )

    add_budget_constraint(m, total_cost, optimization_config)

    # Auxiliary constraints
    add_power_exchange_auxiliary_constraints(
        m, T, P_flow, P_flow_2S, flow_pos_auxi, flow_neg_auxi, flow_u, P_flow_up
    )

    # Objective function
    set_igdt_objective(m, theta)

    optimal_result = solve_and_extract_results(
        m,
        T,
        G_p,
        G_p_cor_abs,
        ESS_pch,
        ESS_pdis,
        ESS_u,
        P_flow,
        flow_pos_auxi,
        flow_neg_auxi,
        G_cost,
        electricity_price,
        theta_u,
        optimization_config,
    )

    return optimal_result
