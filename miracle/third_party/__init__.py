# stdlib
from typing import Any

# miracle relative
from . import imputation_gain  # noqa: F401
from . import imputation_knn  # noqa: F401
from . import imputation_mean  # noqa: F401
from . import imputation_mice  # noqa: F401
from . import imputation_missforest  # noqa: F401


def load_imputer(name: str) -> Any:
    if name == "gain":
        return imputation_gain.GainImputation()
    elif name == "knn":
        return imputation_knn.KNNImputation()
    elif name == "mean":
        return imputation_mean.MeanImputation()
    elif name == "mice":
        return imputation_mice.MiceImputation()
    elif name == "missforest":
        return imputation_missforest.MissForestImputation()
    else:
        raise ValueError(f"unsupported imputation method {name}")
