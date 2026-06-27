import pandas as pd
import numpy as np


def read_ders_excel(filename):
    raw_generators = pd.read_excel(filename, sheet_name="Generators")
    raw_ess = pd.read_excel(filename, sheet_name="ESS")
    raw_prices = pd.read_excel(filename, sheet_name="Prices")
    raw_pv_location = pd.read_excel(filename, sheet_name="PVLocation")
    raw_pv_predictive = pd.read_excel(filename, sheet_name="PVPredictive")

    return raw_generators, raw_ess, raw_prices, raw_pv_location, raw_pv_predictive


def build_generator_data(raw_generators):
    G_node = [raw_generators.loc[i, "Node"] for i in raw_generators.index]
    G_pmax = raw_generators["Pmax"].to_numpy().reshape(-1, 1)
    G_pmin = raw_generators["Pmin"].to_numpy().reshape(-1, 1)
    G_up_limit = raw_generators["RU"].to_numpy()
    G_dn_limit = raw_generators["RD"].to_numpy()
    G_qmax = raw_generators["Qmax"].to_numpy().reshape(-1, 1)
    G_qmin = raw_generators["Qmin"].to_numpy().reshape(-1, 1)
    G_cost = raw_generators["Cost"].to_numpy()

    return G_node, G_pmax, G_pmin, G_qmax, G_qmin, G_cost, G_up_limit, G_dn_limit


def build_ess_data(raw_ess):
    ESS_node = [raw_ess.loc[i, "Node"] for i in raw_ess.index]
    ESS_pmax = raw_ess["Power"].to_numpy().reshape(-1, 1)
    ESS_capacity = raw_ess["Energy"].to_numpy().reshape(-1, 1)
    ESS_Eini = raw_ess["Eini"].to_numpy().reshape(-1, 1)
    ESS_eff = raw_ess["Eff"].to_numpy().reshape(-1, 1)

    return ESS_node, ESS_pmax, ESS_capacity, ESS_Eini, ESS_eff


def build_price_profile(raw_prices, T):
    electricity_price = np.round(raw_prices.loc[:T - 1, "1"].to_numpy(), 2)

    return electricity_price


def build_pv_data(raw_pv_location, raw_pv_predictive, T):
    PV_node = [raw_pv_location.loc[i, "Node"] for i in raw_pv_location.index]

    PV = raw_pv_predictive["1"][:T].to_numpy()
    PV = np.tile(PV, (len(PV_node), 1))

    return PV_node, PV


def processing_DERs_data(filename2, T):
    raw_generators, raw_ess, raw_prices, raw_pv_location, raw_pv_predictive = read_ders_excel(filename2)

    G_node, G_pmax, G_pmin, G_qmax, G_qmin, G_cost, G_up_limit, G_dn_limit = build_generator_data(raw_generators)

    ESS_node, ESS_pmax, ESS_capacity, ESS_Eini, ESS_eff = build_ess_data(raw_ess)

    electricity_price = build_price_profile(raw_prices, T)

    PV_node, PV = build_pv_data(raw_pv_location, raw_pv_predictive, T)

    ders_data = [
        G_node, G_pmax, G_pmin, G_qmax, G_qmin, G_cost,
        electricity_price,
        PV_node, PV,
        G_up_limit, G_dn_limit,
        ESS_node, ESS_pmax, ESS_capacity, ESS_Eini, ESS_eff
    ]

    return ders_data