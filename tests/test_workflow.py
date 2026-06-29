import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

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

from cl_igdt import workflow


def test_load_config_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        workflow.load_config(tmp_path / "config.yaml")


def test_load_config_rejects_empty_file(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Config file is empty"):
        workflow.load_config(config_file)


def test_load_config_rejects_missing_required_section(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump({"time": {"T": 24}}), encoding="utf-8")

    with pytest.raises(KeyError, match="Missing config section"):
        workflow.load_config(config_file)


def test_load_config_reads_valid_config(tmp_path):
    config = {
        "time": {"T": 24},
        "files": {"network_file": "network.xlsx", "ders_file": "ders.xlsx"},
        "network": {"num_nodes": 3},
        "optimization": {"accuracy": 1, "partition_num": 10},
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert workflow.load_config(config_file) == config


def test_solve_optimization_iteration_rejects_failed_optimization(monkeypatch):
    def fake_solve_economic_dispatch_igdt(*args, **kwargs):
        return None

    monkeypatch.setattr(
        workflow.ed, "solve_economic_dispatch_igdt", fake_solve_economic_dispatch_igdt
    )

    with pytest.raises(RuntimeError, match="Optimization failed at iteration 1"):
        workflow.solve_optimization_iteration(
            system_data=None,
            ders_data=None,
            num_nodes=3,
            T=2,
            P_load_Uset={},
            PV_Uset={},
            iteration=1,
            partition_num=10,
            alpha_ini=0,
            accuracy=3,
            network_config={},
            optimization_config={},
        )


def test_solve_optimization_iteration_updates_alpha(monkeypatch):
    def fake_solve_economic_dispatch_igdt(*args, **kwargs):
        return SimpleNamespace(theta_u=np.array([0, 0, 1, 0]))

    monkeypatch.setattr(
        workflow.ed, "solve_economic_dispatch_igdt", fake_solve_economic_dispatch_igdt
    )

    alpha = workflow.solve_optimization_iteration(
        system_data=None,
        ders_data=None,
        num_nodes=3,
        T=2,
        P_load_Uset={},
        PV_Uset={},
        iteration=1,
        partition_num=4,
        alpha_ini=0,
        accuracy=3,
        network_config={},
        optimization_config={},
    )

    assert alpha == 0.2
