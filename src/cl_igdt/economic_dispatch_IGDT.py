from gurobipy import *
import numpy as np


def unpack_system_data(system_data):
    A = system_data[1]
    R_prime = system_data[2] * 0.001
    X_prime = system_data[3] * 0.001
    P_load = system_data[4]

    return A, R_prime, X_prime, P_load


def unpack_ders_data(ders_data, T):
    G_node = ders_data[0]

    G_pmax = np.tile(ders_data[1], (1, T))
    G_pmin = np.tile(ders_data[2], (1, T))
    G_qmax = np.tile(ders_data[3], (1, T))
    G_qmin = np.tile(ders_data[4], (1, T))

    G_cost = ders_data[5]
    electricity_price = ders_data[6]

    PV_node = ders_data[7]
    PV = ders_data[8]

    G_up_limit = ders_data[9]
    G_dn_limit = ders_data[10]

    ESS_node = ders_data[11]
    ESS_pmax = ders_data[12]
    ESS_capacity = ders_data[13]
    ESS_Eini = ders_data[14]
    ESS_eff = ders_data[15]

    return (
        G_node, G_pmax, G_pmin, G_qmax, G_qmin,
        G_cost, electricity_price,
        PV_node, PV,
        G_up_limit, G_dn_limit,
        ESS_node, ESS_pmax, ESS_capacity, ESS_Eini, ESS_eff
    )


def build_opt_model_limits(num_nodes):
    num_nodes = num_nodes-1 #不需要计算首节点

    v0 = 12.66*12.66

    v_up = 1.10 **2 * v0
    v_low = 0.90 **2 * v0

    P_flow_up = 5000
    P_flow_low = -P_flow_up

    return num_nodes, v0, v_up, v_low, P_flow_up, P_flow_low


def create_opt_model():
    m = Model('CL_IGDT')
    m.setParam('OutputFlag', 0)

    return m


def create_generator_variables(m, T, G_node, G_pmin, G_pmax, G_qmin, G_qmax):
    G_p = m.addMVar((len(G_node), T), lb=G_pmin, ub=G_pmax, vtype=GRB.CONTINUOUS, name="G_p")
    G_q = m.addMVar((len(G_node), T), lb=G_qmin, ub=G_qmax, vtype=GRB.CONTINUOUS, name="G_q")

    G_p_cor = m.addMVar((len(G_node), T), lb=-G_pmax * 0.4, ub=G_pmax * 0.4, vtype=GRB.CONTINUOUS, name="G_p_cor")
    G_p_cor_abs = m.addMVar((len(G_node), T), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="G_p_cor_abs")
    G_p_reg = m.addMVar((len(G_node), T), lb=G_pmin, ub=G_pmax, vtype=GRB.CONTINUOUS, name="G_p_reg")

    return G_p, G_q, G_p_cor, G_p_cor_abs, G_p_reg


def create_ess_variables(m, T, ESS_node, ESS_pmax, ESS_capacity):
    ESS_pch = m.addMVar((len(ESS_node), T), lb=0, vtype=GRB.CONTINUOUS, name="ESS_ch")
    ESS_pdis = m.addMVar((len(ESS_node), T), lb=0, vtype=GRB.CONTINUOUS, name="ESS_dis")
    ESS_E = m.addMVar((len(ESS_node), T), lb=ESS_capacity * 0.2, ub=ESS_capacity, vtype=GRB.CONTINUOUS, name="ESS_E")
    ESS_u = m.addMVar((len(ESS_node), T), vtype=GRB.BINARY, name="ESS_u")

    return ESS_pch, ESS_pdis, ESS_E, ESS_u


def create_network_variables(m, num_nodes, T, v_low, v_up, P_flow_low, P_flow_up):
    P_flow = m.addMVar((num_nodes, T), lb=P_flow_low, ub=P_flow_up, name="P_flow")
    P_flow_2S = m.addMVar((num_nodes, T), lb=P_flow_low, ub=P_flow_up, name="P_flow_2S")
    Q_flow = m.addMVar((num_nodes, T), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Q_flow")
    v = m.addMVar((num_nodes, T), lb=v_low, ub=v_up, name="v")

    P_net = m.addMVar((num_nodes, T), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="P_net")
    P_net_2S = m.addMVar((num_nodes, T), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="P_net_2S")
    Q_net = m.addMVar((num_nodes, T), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Q_net")

    return P_flow, P_flow_2S, Q_flow, v, P_net, P_net_2S, Q_net


def create_uncertain_realization_variables(m, num_nodes, T, PV_node):
    P_load_real = m.addMVar((num_nodes, T), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_load_real")
    Q_load = m.addMVar((num_nodes, T), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Q_load")
    PV_real = m.addMVar((len(PV_node), T), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="PV_real")

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
        m.addConstr(
            G_p[:, t] - G_p[:, t - 1] <= G_up_limit,
            name=f"G_p_up_rule_{t}"
        )
        m.addConstr(
            G_p[:, t - 1] - G_p[:, t] <= G_dn_limit,
            name=f"G_p_dn_rule_{t}"
        )


def add_first_stage_ess_constraints(
        m, T, ESS_node,
        ESS_E, ESS_Eini, ESS_eff,
        ESS_pch, ESS_pdis, ESS_u, ESS_pmax
):
    for t in range(T):
        for i in range(len(ESS_node)):
            if t == 0:
                m.addConstr(
                    ESS_E[i, t] == ESS_Eini[i] + ESS_eff[i] * ESS_pch[i, t] - ESS_pdis[i, t] / ESS_eff[i],
                    name=f"ESS_energy_rule_{i}_{t}"
                )
            else:
                m.addConstr(
                    ESS_E[i, t] == ESS_E[i, t - 1] + ESS_eff[i] * ESS_pch[i, t] - ESS_pdis[i, t] / ESS_eff[i],
                    name=f"ESS_energy_rule_{i}_{t}"
                )

            m.addConstr(
                ESS_pch[i, t] <= ESS_u[i, t] * ESS_pmax[i],
                name=f"ESS_ch_limit_{i}_{t}"
            )
            m.addConstr(
                ESS_pdis[i, t] <= (1 - ESS_u[i, t]) * ESS_pmax[i],
                name=f"ESS_dis_limit_{i}_{t}"
            )


def add_first_stage_active_power_balance_constraints(
        m, num_nodes, T,
        G_node, PV_node, ESS_node,
        P_load, PV,
        G_p, ESS_pdis, ESS_pch,
        P_net, P_flow, A
):
    for n in range(num_nodes):
        if n + 1 in G_node:
            for t in range(T):
                i = G_node.index(n + 1)
                m.addConstr(
                    P_net[n, t] == P_load[n, t] + G_p[i, t],
                    name=f"P_net_rule_{n}_{t}"
                )

        elif n + 1 in PV_node:
            for t in range(T):
                i = PV_node.index(n + 1)
                m.addConstr(
                    P_net[n, t] == P_load[n, t] + PV[i, t],
                    name=f"P_net_rule_{n}_{t}"
                )

        elif n + 1 in ESS_node:
            for t in range(T):
                i = ESS_node.index(n + 1)
                m.addConstr(
                    P_net[n, t] == P_load[n, t] + ESS_pdis[i, t] - ESS_pch[i, t],
                    name=f"P_net_rule_{n}_{t}"
                )

        else:
            for t in range(T):
                m.addConstr(
                    P_net[n, t] == P_load[n, t],
                    name=f"P_net_rule_{n}_{t}"
                )

    for t in range(T):
        m.addConstr(
            A.T @ P_flow[:, t] == P_net[:, t],
            name=f"P_Ban_{t}"
        )


def add_second_stage_generator_constraints(
        m, T,
        G_p, G_p_cor, G_p_cor_abs, G_p_reg,
        G_up_limit, G_dn_limit
):
    m.addConstr(G_p_reg == G_p + G_p_cor, name="G_reg_rule")
    m.addConstr(G_p_cor_abs >= G_p_cor, name="G_abs_pos_rule")
    m.addConstr(G_p_cor_abs >= -G_p_cor, name="G_abs_neg_rule")

    for t in range(1, T):
        m.addConstr(
            G_p_reg[:, t] - G_p_reg[:, t - 1] <= G_up_limit,
            name=f"G_reg_up_rule_{t}"
        )
        m.addConstr(
            G_p_reg[:, t - 1] - G_p_reg[:, t] <= G_dn_limit,
            name=f"G_reg_dn_rule_{t}"
        )


def add_second_stage_net_loads_constraints(
        m, num_nodes, T,
        G_node, PV_node, ESS_node,
        P_load_real, Q_load, PV_real,
        G_p_reg, G_q,
        ESS_pdis, ESS_pch,
        P_net_2S, Q_net
):
    for n in range(num_nodes):
        if n + 1 in G_node:
            for t in range(T):
                i = G_node.index(n + 1)
                m.addConstr(
                    P_net_2S[n, t] == P_load_real[n, t] + G_p_reg[i, t],
                    name=f"P_net_2S_rule_{n}_{t}"
                )
                m.addConstr(
                    Q_net[n, t] == Q_load[n, t] + G_q[i, t],
                    name=f"Q_net_rule_{n}_{t}"
                )

        elif n + 1 in PV_node:
            for t in range(T):
                i = PV_node.index(n + 1)
                m.addConstr(
                    P_net_2S[n, t] == P_load_real[n, t] + PV_real[i, t],
                    name=f"P_net_2S_rule_{n}_{t}"
                )
                m.addConstr(
                    Q_net[n, t] == Q_load[n, t],
                    name=f"Q_net_rule_{n}_{t}"
                )

        elif n + 1 in ESS_node:
            for t in range(T):
                i = ESS_node.index(n + 1)
                m.addConstr(
                    P_net_2S[n, t] == P_load_real[n, t] + ESS_pdis[i, t] - ESS_pch[i, t],
                    name=f"P_net_2S_rule_{n}_{t}"
                )
                m.addConstr(
                    Q_net[n, t] == Q_load[n, t],
                    name=f"Q_net_rule_{n}_{t}"
                )

        else:
            for t in range(T):
                m.addConstr(
                    P_net_2S[n, t] == P_load_real[n, t],
                    name=f"P_net_2S_rule_{n}_{t}"
                )
                m.addConstr(
                    Q_net[n, t] == Q_load[n, t],
                    name=f"Q_net_rule_{n}_{t}"
                )


def add_second_stage_network_constraints(
        m, T,
        A, R_prime, X_prime, v0,
        P_flow_2S, Q_flow,
        P_net_2S, Q_net, v
):
    for t in range(T):
        m.addConstr(
            A.T @ P_flow_2S[:, t] == P_net_2S[:, t],
            name=f"P_Ban_2S_{t}"
        )
        m.addConstr(
            A.T @ Q_flow[:, t] == Q_net[:, t],
            name=f"Q_Ban_{t}"
        )
        m.addConstr(
            v[:, t] == v0 + (R_prime @ P_net_2S[:, t] + X_prime @ Q_net[:, t]),
            name=f"v_rule_{t}"
        )


def add_igdt_partition_constraints(
        m, partition_num, iter, α_ini,
        theta, theta_s, theta_u
):
    for i in range(partition_num):
        m.addConstr(
            theta_s[i] >= (α_ini + 10 ** (-iter) * i) * theta_u[i],
            name=f"theta_s_limt1_{i}"
        )
        m.addConstr(
            theta_s[i] <= (α_ini + 10 ** (-iter) * (i + 1)) * theta_u[i],
            name=f"theta_s_limt2_{i}"
        )

    m.addConstr(
        quicksum(theta_u[i] for i in range(partition_num)) == 1,
        name="obj_u_rule"
    )
    m.addConstr(
        quicksum(theta_s[i] for i in range(partition_num)) == theta,
        name="theta_rule"
    )


def add_uncertainty_realization_constraints(
        m, partition_num, iter, α_ini,
        P_load_Uset, PV_Uset,
        P_load_real, Q_load, PV_real,
        theta_u
):
    m.addConstr(
        P_load_real == quicksum(
            P_load_Uset[round(10 ** (-iter) * i + α_ini, iter)] * theta_u[i]
            for i in range(partition_num)
        ),
        name="Load_Uset_rule"
    )

    m.addConstr(
        Q_load == P_load_real * 0.9,
        name="Reactive_Load_rule"
    )

    m.addConstr(
        PV_real == quicksum(
            PV_Uset[round(10 ** (-iter) * i + α_ini, iter)] * theta_u[i]
            for i in range(partition_num)
        ),
        name="PV_Uset_rule"
    )


def build_total_cost(
        T,
        G_p, G_p_cor_abs, P_flow,
        flow_pos_auxi, flow_neg_auxi,
        G_cost, electricity_price
):
    total_cost = (
            quicksum(G_p[:, t] for t in range(T)) @ G_cost
            + quicksum(G_p_cor_abs[:, t] for t in range(T)) @ G_cost * 1.5
            + quicksum(P_flow[0, t] * electricity_price[t] for t in range(T))
            + quicksum(flow_pos_auxi[t] * electricity_price[t] * 1.4 for t in range(T))
            - quicksum(flow_neg_auxi[t] * electricity_price[t] * 0.8 for t in range(T))
    )

    return total_cost


def add_budget_constraint(m, total_cost):
    m.addConstr(
        total_cost <= 223458.23 * 1.3,
        name="Budget_limit"
    )


def add_power_exchange_auxiliary_constraints(
        m, T,
        P_flow, P_flow_2S,
        flow_pos_auxi, flow_neg_auxi, flow_u,
        P_flow_up
):
    for t in range(T):
        m.addConstr(
            flow_pos_auxi[t] - flow_neg_auxi[t] == P_flow_2S[0, t] - P_flow[0, t],
            name="Flow_diff_rule"
        )
        m.addConstr(
            flow_pos_auxi[t] <= flow_u[t] * P_flow_up * 2,
            name="Flow_abs_pos_rule"
        )
        m.addConstr(
            flow_neg_auxi[t] <= (1 - flow_u[t]) * P_flow_up * 2,
            name="Flow_abs_neg_rule"
        )


def set_igdt_objective(m, theta):
    m.setObjective(theta, GRB.MAXIMIZE)


def solve_and_extract_results(
        m, T,
        G_p, G_p_cor_abs,
        ESS_pch, ESS_pdis, ESS_u,
        P_flow,
        flow_pos_auxi, flow_neg_auxi,
        G_cost, electricity_price,
        theta_u
):
    m.optimize()

    if m.status == GRB.OPTIMAL:
        global G_p_x, ESS_pch_x, ESS_pdis_x, ESS_u_x, P_flow_x

        G_p_x = G_p.x
        ESS_pch_x = ESS_pch.x
        ESS_pdis_x = ESS_pdis.x
        ESS_u_x = ESS_u.x
        P_flow_x = P_flow.x

        total_cost_spent = (
            quicksum(G_p.x[:, t] for t in range(T)) @ G_cost
            + quicksum(G_p_cor_abs.x[:, t] for t in range(T)) @ G_cost * 1.5
            + quicksum(P_flow.x[0, t] * electricity_price[t] for t in range(T))
            + quicksum(flow_pos_auxi.x[t] * electricity_price[t] * 1.4 for t in range(T))
            - quicksum(flow_neg_auxi.x[t] * electricity_price[t] * 0.8 for t in range(T))
        )

        first_stage_cost = (
            quicksum(G_p.x[:, t] for t in range(T)) @ G_cost
            + quicksum(P_flow.x[0, t] * electricity_price[t] for t in range(T))
        )

        return theta_u.x, round(total_cost_spent.getValue(), 2), round(first_stage_cost.getValue(), 2)

    else:
        return None, None


def economic_dispatch_IGDT(System_Data, DERs_Data, num_nodes, T, P_load_Uset, PV_Uset, iter, partition_num, α_ini):
## ---------  Parameters --------------
    A, R_prime, X_prime, P_load = unpack_system_data(System_Data)

    (
        G_node, G_pmax, G_pmin, G_qmax, G_qmin,
        G_cost, electricity_price,
        PV_node, PV,
        G_up_limit, G_dn_limit,
        ESS_node, ESS_pmax, ESS_capacity, ESS_Eini, ESS_eff
    ) = unpack_ders_data(DERs_Data, T)

    num_nodes, v0, v_up, v_low, P_flow_up, P_flow_low = build_opt_model_limits(num_nodes)


## ---------  Decision Variables --------------
    m = create_opt_model()

    G_p, G_q, G_p_cor, G_p_cor_abs, G_p_reg = create_generator_variables(
        m, T, G_node, G_pmin, G_pmax, G_qmin, G_qmax
    )

    ESS_pch, ESS_pdis, ESS_E, ESS_u = create_ess_variables(
        m, T, ESS_node, ESS_pmax, ESS_capacity
    )

    P_flow, P_flow_2S, Q_flow, v, P_net, P_net_2S, Q_net = create_network_variables(
        m, num_nodes, T, v_low, v_up, P_flow_low, P_flow_up
    )

    P_load_real, Q_load, PV_real = create_uncertain_realization_variables(
        m, num_nodes, T, PV_node
    )

    flow_pos_auxi, flow_neg_auxi, flow_u = create_power_exchange_auxiliary_variables(
        m, T
    )

    theta, theta_s, theta_u = create_igdt_obj_variables(
        m, partition_num
    )

## ---------  First-stage constraints --------------
    add_first_stage_generator_constraints(
        m, T, G_p, G_up_limit, G_dn_limit
    )

    add_first_stage_ess_constraints(
        m, T, ESS_node,
        ESS_E, ESS_Eini, ESS_eff,
        ESS_pch, ESS_pdis, ESS_u, ESS_pmax
    )

    add_first_stage_active_power_balance_constraints(
        m, num_nodes, T,
        G_node, PV_node, ESS_node,
        P_load, PV,
        G_p, ESS_pdis, ESS_pch,
        P_net, P_flow, A
    )

## ---------  Second-stage constraints --------------
    add_second_stage_generator_constraints(
        m, T,
        G_p, G_p_cor, G_p_cor_abs, G_p_reg,
        G_up_limit, G_dn_limit
    )

    add_second_stage_net_loads_constraints(
        m, num_nodes, T,
        G_node, PV_node, ESS_node,
        P_load_real, Q_load, PV_real,
        G_p_reg, G_q,
        ESS_pdis, ESS_pch,
        P_net_2S, Q_net
    )

    add_second_stage_network_constraints(
        m, T,
        A, R_prime, X_prime, v0,
        P_flow_2S, Q_flow,
        P_net_2S, Q_net, v
    )

##------  Uncertainty set and stuff--------------##
    ## Partition of objective function
    add_igdt_partition_constraints(
        m, partition_num, iter, α_ini,
        theta, theta_s, theta_u
    )

    ## Partition of uncertainty set
    add_uncertainty_realization_constraints(
        m, partition_num, iter, α_ini,
        P_load_Uset, PV_Uset,
        P_load_real, Q_load, PV_real,
        theta_u
    )

    ## Budget limit
    total_cost = build_total_cost(
        T,
        G_p, G_p_cor_abs, P_flow,
        flow_pos_auxi, flow_neg_auxi,
        G_cost, electricity_price
    )

    add_budget_constraint(m, total_cost)

    ## Auxiliary constraints
    add_power_exchange_auxiliary_constraints(
        m, T,
        P_flow, P_flow_2S,
        flow_pos_auxi, flow_neg_auxi, flow_u,
        P_flow_up
    )

## ---------  Objective function --------------
    set_igdt_objective(m, theta)

    return solve_and_extract_results(
        m, T,
        G_p, G_p_cor_abs,
        ESS_pch, ESS_pdis, ESS_u,
        P_flow,
        flow_pos_auxi, flow_neg_auxi,
        G_cost, electricity_price,
        theta_u
    )

def get_decision_variables():
    decision_variables = [G_p_x, ESS_pch_x, ESS_pdis_x, ESS_u_x, P_flow_x]
    return decision_variables