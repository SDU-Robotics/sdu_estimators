from multiprocessing.managers import Value

from sdu_estimators._sdu_estimators import GradientEstimator

from sdu_estimators._sdu_estimators import Kreisselmeier_1x2
from sdu_estimators._sdu_estimators import Kreisselmeier_1x3
from sdu_estimators._sdu_estimators import RegressorExtension_1x2
from sdu_estimators._sdu_estimators import RegressorExtension_1x3

from sdu_estimators._sdu_estimators import DREM_1x2
from sdu_estimators._sdu_estimators import DREM_1x3

# from sdu_estimators._sdu_estimators import Kreisselmeier
# from sdu_estimators._sdu_estimators import RegressorExtension

# Export the version given in project metadata
from importlib import metadata

__version__ = metadata.version(__package__)
del metadata

def DREM(dt, gamma, theta_init, KRE, r = 1.):
    # assert gamma.size() == theta_init.size()
    DIM_P = len(theta_init)

    match DIM_P:
        case 2:
            # print()
            return DREM_1x2(dt, gamma, theta_init, KRE, r)

        case 3:
            return DREM_1x3(dt, gamma, theta_init, KRE, r)

        case _:
            raise ValueError(f"Wrong p-dimension. {DIM_P} is not supported.")