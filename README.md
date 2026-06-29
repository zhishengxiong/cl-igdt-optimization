# Uncertainty-Aware Energy Scheduling

A Python optimization project for uncertainty-aware energy asset scheduling, based on the CL-IGDT framework proposed in:

> Zhisheng Xiong, Bo Zeng, Peter Palensky, Pedro P. Vergara,
> "Optimal operation of distribution networks under asymmetric renewable energy and load demand uncertainties",
> Sustainable Energy, Grids and Networks, 2026.
> DOI: 10.1016/j.segan.2026.102289

This repository implements a confidence-level-based information gap decision theory (CL-IGDT) workflow for uncertainty-aware energy asset scheduling. It includes data preprocessing, uncertainty-set construction, optimization model construction, unit tests, and a reproducible 69-bus demo case.

## What This Project Shows

- Uncertainty-aware optimization for energy asset scheduling.
- Confidence-level information gap decision theory (CL-IGDT).
- Data-driven uncertainty-set construction from historical samples.
- Base scheduling and corrective recourse decisions.
- Operational constraints for flexible energy assets and network feasibility.
- Package-style Python structure with unit tests and GitHub Actions CI.

## Repository Structure

```text
cl-igdt-optimization/
|-- .editorconfig
|-- configs/
|   `-- config.yaml
|-- data/
|   |-- IEEE69.xlsx
|   |-- DERs_Data_69.xlsx
|   |-- historical_data_demand_69.xlsx
|   |-- historical_data_PV_69.xlsx
|   `-- cached uncertainty-set files
|-- examples/
|   `-- run_case.py
|-- src/
|   `-- cl_igdt/
|       |-- workflow.py
|       |-- system_data_preprocessing.py
|       |-- ders_data_preprocessing.py
|       |-- demand_uncertainty_set.py
|       |-- pv_uncertainty_set.py
|       `-- economic_dispatch_igdt.py
|-- tests/
|-- docs/
|   `-- optimization_model.md
|-- CITATION.cff
|-- LICENSE
|-- pyproject.toml
`-- README.md
```

## Data Sources

The load and PV data in the included 69-bus case were generated using the method from:

> H. Shengren, P. P. Vergara, E. M. Salazar Duque, P. Palensky,
> "Optimal energy system scheduling using a constraint-aware reinforcement learning algorithm",
> International Journal of Electrical Power & Energy Systems, 152, 109230, 2023.
> DOI: 10.1016/j.ijepes.2023.109230

The IEEE 69-bus system data follow:

> O. D. Montoya, C. A. Ramos-Paja, L. F. Grisales-Norena,
> "An efficient methodology for locating and sizing PV generators in radial distribution networks using a mixed-integer conic relaxation",
> Mathematics, 10(15), 2626, 2022.

## Requirements

- Python 3.10 or newer
- Gurobi Optimizer
- A valid Gurobi license
- Python packages listed in `pyproject.toml`

The optimization model is written with the Gurobi Python API (`gurobipy`), so Gurobi is a required dependency.

## Installation

Create and activate a Python environment, then install the project in editable mode:

```bash
pip install -e ".[test]"
```

If you only want runtime dependencies:

```bash
pip install -e .
```

## Run the 69-Bus Demo

From the repository root:

```bash
python examples/run_case.py
```

The example loads:

- `configs/config.yaml`
- network and DER data from `data/`
- cached or newly generated demand and PV uncertainty sets

The workflow iteratively searches the confidence level and prints the optimal confidence level at each iteration.

## Configuration

The main configuration file is:

```text
configs/config.yaml
```

It controls:

- time horizon
- input file names
- network size and base voltage
- IGDT search accuracy and partition count
- voltage and feeder-flow limits
- generator recourse limits
- ESS operating assumptions
- budget and second-stage cost assumptions

## Tests

Run:

```bash
pytest
```

The tests cover:

- configuration validation
- network-data preprocessing
- DER-data preprocessing
- demand and PV uncertainty-set helpers
- selected optimization-model utilities
- workflow-level behavior using mocked optimization results

## Code Quality

Run linting, format checking, and tests:

```bash
ruff check .
ruff format --check .
pytest
```

GitHub Actions runs Ruff linting, Ruff format checking, and pytest on push and pull request.

## Optimization Notes

For a compact technical explanation of the optimization workflow, see:

```text
docs/optimization_model.md
```

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Citation

If you use this code, please cite the software metadata in `CITATION.cff` and the associated paper:

```bibtex
@article{xiong2026cligdt,
  title = {Optimal operation of distribution networks under asymmetric renewable energy and load demand uncertainties},
  author = {Xiong, Zhisheng and Zeng, Bo and Palensky, Peter and Vergara, Pedro P.},
  journal = {Sustainable Energy, Grids and Networks},
  year = {2026},
  doi = {10.1016/j.segan.2026.102289}
}
```
