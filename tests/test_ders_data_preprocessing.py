import numpy as np
import pandas as pd
import pytest

from cl_igdt import ders_data_preprocessing as pder


def write_minimal_ders_file(path, price_rows=3, pv_rows=3, include_all_sheets=True):
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(
            {
                pder.NODE_COLUMN: [2],
                pder.PMAX_COLUMN: [100.0],
                pder.PMIN_COLUMN: [10.0],
                pder.QMAX_COLUMN: [50.0],
                pder.QMIN_COLUMN: [-50.0],
                pder.RU_COLUMN: [20.0],
                pder.RD_COLUMN: [20.0],
                pder.COST_COLUMN: [30.0],
            }
        ).to_excel(writer, sheet_name=pder.GENERATORS_SHEET, index=False)

        if include_all_sheets:
            pd.DataFrame(
                {
                    pder.NODE_COLUMN: [3],
                    pder.POWER_COLUMN: [40.0],
                    pder.ENERGY_COLUMN: [80.0],
                    pder.EINI_COLUMN: [20.0],
                    pder.EFF_COLUMN: [0.95],
                }
            ).to_excel(writer, sheet_name=pder.ESS_SHEET, index=False)
            pd.DataFrame(
                {pder.PRICE_COLUMN: np.arange(price_rows, dtype=float)}
            ).to_excel(writer, sheet_name=pder.PRICES_SHEET, index=False)
            pd.DataFrame({pder.NODE_COLUMN: [4, 5]}).to_excel(
                writer, sheet_name=pder.PV_LOCATION_SHEET, index=False
            )
            pd.DataFrame({pder.PV_COLUMN: np.arange(pv_rows, dtype=float)}).to_excel(
                writer, sheet_name=pder.PV_PREDICTIVE_SHEET, index=False
            )


def test_read_ders_data_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="DERs data file not found"):
        pder.read_ders_data(tmp_path / "missing.xlsx")


def test_read_ders_data_rejects_missing_sheet(tmp_path):
    ders_file = tmp_path / "ders.xlsx"
    write_minimal_ders_file(ders_file, include_all_sheets=False)

    with pytest.raises(ValueError, match="Missing sheet 'ESS'"):
        pder.read_ders_data(ders_file)


def test_build_price_profile_rejects_short_profile():
    raw_prices = pd.DataFrame({pder.PRICE_COLUMN: [1.0, 2.0]})

    with pytest.raises(ValueError, match="Price profile length 2 is shorter than T=3"):
        pder.build_price_profile(raw_prices, T=3)


def test_build_pv_data_rejects_short_profile():
    raw_pv_location = pd.DataFrame({pder.NODE_COLUMN: [4]})
    raw_pv_predictive = pd.DataFrame({pder.PV_COLUMN: [0.0, 1.0]})

    with pytest.raises(
        ValueError, match="PV predictive profile length 2 is shorter than T=3"
    ):
        pder.build_pv_data(raw_pv_location, raw_pv_predictive, T=3)


def test_build_ders_data_from_minimal_excel_file(tmp_path):
    ders_file = tmp_path / "ders.xlsx"
    write_minimal_ders_file(ders_file, price_rows=4, pv_rows=4)

    ders_data = pder.build_ders_data(ders_file, T=3)

    assert ders_data.G_node == [2]
    assert ders_data.ESS_node == [3]
    assert ders_data.PV_node == [4, 5]

    assert ders_data.G_pmax.shape == (1, 1)
    assert ders_data.ESS_pmax.shape == (1, 1)
    assert ders_data.PV.shape == (2, 3)

    np.testing.assert_array_equal(
        ders_data.electricity_price, np.array([0.0, 1.0, 2.0])
    )
    np.testing.assert_array_equal(
        ders_data.PV,
        np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]),
    )
