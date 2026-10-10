# credit-risk-model-monitoring

Automated credit-risk model monitoring: a probability-of-default (PD) model on LendingClub loans, monthly
PSI/CSI, Gini/KS and calibration checks with red/amber/green alerts, auto-generated reports and
Tableau-ready dashboard data, orchestrated by Airflow.

> Work in progress - full write-up lands with the integration PR. Interfaces: [docs/CONTRACTS.md](docs/CONTRACTS.md).

```bash
make setup      # venv + pinned deps
make test lint  # fixtures only
make demo       # small-fixture end-to-end
make e2e        # full LendingClub run (needs ~/.kaggle/access_token)
```
