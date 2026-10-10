import pandas as pd
import pytest

from common import schemas
from common.config import CSI_FEATURES


def test_results_validation_rejects_bad_rag():
    row = dict.fromkeys(schemas.RESULT_COLUMNS, 0)
    row.update(month="2014-01", metric="psi_score", segment="ALL", direction="higher_worse", rag="PURPLE")
    with pytest.raises(ValueError):
        schemas.validate_results(pd.DataFrame([row]))


def test_scored_requires_csi_features():
    df = pd.DataFrame({c: [0] for c in schemas.SCORED_CORE_COLUMNS})
    df["outcome_observable"] = False
    df["default_flag"] = pd.array([pd.NA], dtype="Int8")
    with pytest.raises(ValueError, match="missing"):
        schemas.validate_scored(df)
    for f in CSI_FEATURES:
        df[f] = 1
    schemas.validate_scored(df)
