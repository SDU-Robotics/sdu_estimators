from sdu_estimators._sdu_estimators import add_one
from sdu_estimators._sdu_estimators import GradientEstimator

# Export the version given in project metadata
from importlib import metadata

__version__ = metadata.version(__package__)
del metadata


def one_plus_one():
    return add_one(1)
