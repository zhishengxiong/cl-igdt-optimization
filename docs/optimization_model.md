# Uncertainty-Aware Optimization Algorithm

This document explains the optimization algorithm behind the CL-IGDT demo in a way that is relevant beyond distribution-network operation. The original research application is power-system operation under renewable and load uncertainty, but the transferable core is an uncertainty-aware optimization workflow for scheduling, recourse, and risk control.

For market clearing, aggregators, retailers, or trading teams, the same modelling pattern appears in:

- day-ahead scheduling and/or intraday adjustment of flexible assets
- battery, generator, PV, demand-response, and portfolio dispatch
- bid-volume decisions under forecast uncertainty
- imbalance-risk management
- cost-risk trade-offs for energy portfolios

The code therefore demonstrates not only a power-grid model, but also practical optimization skills that transfer to trading, aggregation, and bidding algorithms.

## What This Project Demonstrates

- Formulating a real energy decision problem as a mixed-integer linear program (MILP).
- Separating base decisions from corrective actions, similar to day-ahead planning followed by real-time or intraday adjustment.
- Building data-driven uncertainty sets from historical samples instead of assuming a fixed probability distribution.
- Maximizing robustness under a cost budget, which is close to choosing how much uncertainty a trading or aggregation strategy can tolerate before becoming too expensive.
- Encoding operational constraints for physical assets, including ramping, storage dynamics, import/export logic, and network feasibility.
- Implementing the full workflow in modular Python: data preprocessing, uncertainty-set construction, optimization, iterative search, and tests.

## Energy-Market Interpretation

| Model component       | Research wording | Market / aggregation interpretation |
|-----------------------| --- | --- |
| First-stage dispatch  | Planned generator, ESS, and feeder operation | Day-ahead schedule, base bid, or planned asset position |
| Second-stage recourse | Corrective generation and power-exchange adjustment | Intraday rebidding, balancing action, or imbalance correction |
| Load uncertainty      | Unknown demand realization | Customer demand / portfolio volume uncertainty |
| PV uncertainty        | Unknown renewable generation | Renewable forecast error |
| ESS constraints       | Battery charge, discharge, and state of charge | Flexible asset scheduling |
| Budget constraint     | Allowed operation-cost deviation | Risk appetite or maximum acceptable imbalance/recourse cost |
| `alpha`               | Confidence level / uncertainty tolerance | Robustness level of the schedule or bid |
| Network constraints   | Voltage and branch-flow feasibility | Operational feasibility constraints for assets and delivery limits |
| Binary selectors      | Partition and operating-mode decisions | Discrete regime, mode, or bid-structure choices |

The distribution-network equations are the domain-specific layer. The algorithmic layer is more general: decide a base schedule, protect it against uncertain realizations, and use limited corrective actions while respecting a cost/risk budget.

## Problem Type

The implemented model is a multi-period, two-stage, mixed-integer linear optimization problem with a data-driven uncertainty set.

In compact form:

```text
maximize        robustness: confidence level

subject to      base schedule constraints
                uncertain realization selection
                corrective-action constraints
                asset and network feasibility constraints
                total cost <= allowed budget
                binary mode and partition decisions
```

This is different from a pure deterministic model. The model does not only ask "what is the cheapest schedule for one forecast?". It asks:

```text
How much forecast uncertainty can this schedule survive
while keeping the total expected operating and correction cost
within an acceptable budget?
```

That question is directly relevant to bidding and aggregation: a bid or dispatch plan is valuable only if it remains feasible and economically acceptable when actual demand, renewable generation, or portfolio behavior deviates from the forecast.

## Workflow

The workflow in `src/cl_igdt/workflow.py` has four main steps:

1. Load configuration from `configs/config.yaml`.
2. Build system, asset, and price data from file inputs.
3. Construct or load IDM-based uncertainty sets for demand and PV.
4. Solve the optimization model and update the CL-IGDT confidence-level search.

The CL-IGDT search refines `alpha` digit by digit. At each iteration, the current confidence interval is split into candidate partitions. A binary selector chooses one partition, and the selected value becomes the basis for the next iteration.

This is similar to searching for the highest acceptable robustness level of a schedule under a fixed risk/cost tolerance.

For more details, please refer to the original paper:

```text
Zhisheng Xiong, Bo Zeng, Peter Palensky, and Pedro P. Vergara, “Optimal operation of distribution networks under asymmetric renewable energy and load demand uncertainties,” Sustainable Energy, Grids and Networks, 2026.
```