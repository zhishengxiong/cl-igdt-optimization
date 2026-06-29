import sys
import types
from types import SimpleNamespace

import numpy as np

try:
    import gurobipy  # noqa: F401
except ImportError:
    fake_gurobipy = types.ModuleType("gurobipy")
    fake_gurobipy.GRB = SimpleNamespace(
        CONTINUOUS=0,
        BINARY=1,
        INFINITY=1e100,
        OPTIMAL=2,
        MAXIMIZE=1,
    )
    fake_gurobipy.Model = object
    fake_gurobipy.quicksum = sum
    sys.modules["gurobipy"] = fake_gurobipy

from cl_igdt import economic_dispatch_igdt as ed


def test_unpack_system_data_applies_fixed_formula_scaling():
    system_data = SimpleNamespace(
        A=np.array([[1.0, -1.0]]),
        R_prime=np.array([[2.0]]),
        X_prime=np.array([[3.0]]),
        P_load=np.array([[-10.0, -12.0]]),
    )

    A, R_prime, X_prime, P_load = ed.unpack_system_data(system_data)

    np.testing.assert_array_equal(A, system_data.A)
    np.testing.assert_allclose(R_prime, np.array([[0.002]]))
    np.testing.assert_allclose(X_prime, np.array([[0.003]]))
    np.testing.assert_array_equal(P_load, system_data.P_load)


def test_unpack_ders_data_tiles_generator_limits():
    ders_data = SimpleNamespace(
        G_node=[2],
        G_pmax=np.array([[100.0]]),
        G_pmin=np.array([[10.0]]),
        G_qmax=np.array([[50.0]]),
        G_qmin=np.array([[-50.0]]),
        G_cost=np.array([30.0]),
        electricity_price=np.array([1.0, 2.0, 3.0]),
        PV_node=[3],
        PV=np.array([[0.0, 1.0, 2.0]]),
        G_up_limit=np.array([20.0]),
        G_dn_limit=np.array([20.0]),
        ESS_node=[4],
        ESS_pmax=np.array([[40.0]]),
        ESS_capacity=np.array([[80.0]]),
        ESS_Eini=np.array([[20.0]]),
        ESS_eff=np.array([[0.95]]),
    )

    unpacked = ed.unpack_ders_data(ders_data, T=3)
    G_pmax = unpacked[1]
    G_pmin = unpacked[2]

    assert G_pmax.shape == (1, 3)
    assert G_pmin.shape == (1, 3)
    np.testing.assert_array_equal(G_pmax, np.array([[100.0, 100.0, 100.0]]))
    np.testing.assert_array_equal(G_pmin, np.array([[10.0, 10.0, 10.0]]))


def test_build_opt_model_limits():
    network_config = {"base_voltage_kv": 10.0}
    optimization_config = {
        "voltage_upper_limit": 1.1,
        "voltage_lower_limit": 0.9,
        "p_flow_upper_limit": 5000,
    }

    limits = ed.build_opt_model_limits(
        num_nodes=4,
        network_config=network_config,
        optimization_config=optimization_config,
    )

    assert limits[0] == 3
    assert limits[1] == 100.0
    assert limits[4:] == (5000, -5000)
    np.testing.assert_allclose(limits[2], 121.0)
    np.testing.assert_allclose(limits[3], 81.0)
