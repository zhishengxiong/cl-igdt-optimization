"""Build DERs and electricity price data for the optimization model.

The input file follows the original research data schema.
Some column names are kept unchanged for reproducibility.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


GENERATORS_SHEET = "Generators"
ESS_SHEET = "ESS"
PRICES_SHEET = "Prices"
PV_LOCATION_SHEET = "PVLocation"
PV_PREDICTIVE_SHEET = "PVPredictive"

NODE_COLUMN = "Node"
PMAX_COLUMN = "Pmax"
PMIN_COLUMN = "Pmin"
QMAX_COLUMN = "Qmax"
QMIN_COLUMN = "Qmin"
RU_COLUMN = "RU"
RD_COLUMN = "RD"
COST_COLUMN = "Cost"

POWER_COLUMN = "Power"
ENERGY_COLUMN = "Energy"
EINI_COLUMN = "Eini"
EFF_COLUMN = "Eff"

PRICE_COLUMN = "1"
PV_COLUMN = "1"


@dataclass
class DERsData:
    G_node: list
    G_pmax: np.ndarray
    G_pmin: np.ndarray
    G_qmax: np.ndarray
    G_qmin: np.ndarray
    G_cost: np.ndarray
    electricity_price: np.ndarray
    PV_node: list
    PV: np.ndarray
    G_up_limit: np.ndarray
    G_dn_limit: np.ndarray
    ESS_node: list
    ESS_pmax: np.ndarray
    ESS_capacity: np.ndarray
    ESS_Eini: np.ndarray
    ESS_eff: np.ndarray


def read_ders_data(ders_file):
    if not ders_file.exists():
        raise FileNotFoundError(f"DERs data file not found: {ders_file}")

    required_sheets = [
        GENERATORS_SHEET,
        ESS_SHEET,
        PRICES_SHEET,
        PV_LOCATION_SHEET,
        PV_PREDICTIVE_SHEET,
    ]

    available_sheets = pd.ExcelFile(ders_file).sheet_names

    for sheet in required_sheets:
        if sheet not in available_sheets:
            raise ValueError(f"Missing sheet '{sheet}' in DERs data file: {ders_file}")

    raw_generators = pd.read_excel(ders_file, sheet_name=GENERATORS_SHEET)
    raw_ess = pd.read_excel(ders_file, sheet_name=ESS_SHEET)
    raw_prices = pd.read_excel(ders_file, sheet_name=PRICES_SHEET)
    raw_pv_location = pd.read_excel(ders_file, sheet_name=PV_LOCATION_SHEET)
    raw_pv_predictive = pd.read_excel(ders_file, sheet_name=PV_PREDICTIVE_SHEET)

    return raw_generators, raw_ess, raw_prices, raw_pv_location, raw_pv_predictive


def build_generator_data(raw_generators):
    G_node = raw_generators[NODE_COLUMN].tolist()

    G_pmax = raw_generators[PMAX_COLUMN].to_numpy().reshape(-1, 1)
    G_pmin = raw_generators[PMIN_COLUMN].to_numpy().reshape(-1, 1)
    G_qmax = raw_generators[QMAX_COLUMN].to_numpy().reshape(-1, 1)
    G_qmin = raw_generators[QMIN_COLUMN].to_numpy().reshape(-1, 1)

    G_cost = raw_generators[COST_COLUMN].to_numpy()
    G_up_limit = raw_generators[RU_COLUMN].to_numpy()
    G_dn_limit = raw_generators[RD_COLUMN].to_numpy()

    return (
        G_node,
        G_pmax,
        G_pmin,
        G_qmax,
        G_qmin,
        G_cost,
        G_up_limit,
        G_dn_limit,
    )


def build_ess_data(raw_ess):
    ESS_node = raw_ess[NODE_COLUMN].tolist()

    ESS_pmax = raw_ess[POWER_COLUMN].to_numpy().reshape(-1, 1)
    ESS_capacity = raw_ess[ENERGY_COLUMN].to_numpy().reshape(-1, 1)
    ESS_Eini = raw_ess[EINI_COLUMN].to_numpy().reshape(-1, 1)
    ESS_eff = raw_ess[EFF_COLUMN].to_numpy().reshape(-1, 1)

    return ESS_node, ESS_pmax, ESS_capacity, ESS_Eini, ESS_eff


def build_price_profile(raw_prices, T):
    if len(raw_prices) < T:
        raise ValueError(
            f"Price profile length {len(raw_prices)} is shorter than T={T}"
        )

    electricity_price = np.round(
        raw_prices.loc[: T - 1, PRICE_COLUMN].to_numpy(),
        2,
    )

    return electricity_price


def build_pv_data(raw_pv_location, raw_pv_predictive, T):
    if len(raw_pv_predictive) < T:
        raise ValueError(
            f"PV predictive profile length {len(raw_pv_predictive)} is shorter than T={T}"
        )

    PV_node = raw_pv_location[NODE_COLUMN].tolist()

    PV = raw_pv_predictive[PV_COLUMN][:T].to_numpy()
    PV = np.tile(PV, (len(PV_node), 1))

    return PV_node, PV


def build_ders_data(ders_file, T):
    (
        raw_generators,
        raw_ess,
        raw_prices,
        raw_pv_location,
        raw_pv_predictive,
    ) = read_ders_data(ders_file)

    (
        G_node,
        G_pmax,
        G_pmin,
        G_qmax,
        G_qmin,
        G_cost,
        G_up_limit,
        G_dn_limit,
    ) = build_generator_data(raw_generators)

    ESS_node, ESS_pmax, ESS_capacity, ESS_Eini, ESS_eff = build_ess_data(raw_ess)

    electricity_price = build_price_profile(raw_prices, T)

    PV_node, PV = build_pv_data(raw_pv_location, raw_pv_predictive, T)

    ders_data = DERsData(
        G_node=G_node,
        G_pmax=G_pmax,
        G_pmin=G_pmin,
        G_qmax=G_qmax,
        G_qmin=G_qmin,
        G_cost=G_cost,
        electricity_price=electricity_price,
        PV_node=PV_node,
        PV=PV,
        G_up_limit=G_up_limit,
        G_dn_limit=G_dn_limit,
        ESS_node=ESS_node,
        ESS_pmax=ESS_pmax,
        ESS_capacity=ESS_capacity,
        ESS_Eini=ESS_Eini,
        ESS_eff=ESS_eff,
    )

    return ders_data
