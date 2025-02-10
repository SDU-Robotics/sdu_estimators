# from multiprocessing.managers import Value
import numpy as np
from sdu_estimators._sdu_estimators import GradientEstimator_1x1
from sdu_estimators._sdu_estimators import GradientEstimator_1x2
from sdu_estimators._sdu_estimators import GradientEstimator_1x3
from sdu_estimators._sdu_estimators import GradientEstimator_3x6
# from sdu_estimators._sdu_estimators import GradientEstimator_1x4

from sdu_estimators._sdu_estimators import Kreisselmeier_1x1
from sdu_estimators._sdu_estimators import Kreisselmeier_1x2
from sdu_estimators._sdu_estimators import Kreisselmeier_1x3
from sdu_estimators._sdu_estimators import Kreisselmeier_3x6
from sdu_estimators._sdu_estimators import RegressorExtension_1x1
from sdu_estimators._sdu_estimators import RegressorExtension_1x2
from sdu_estimators._sdu_estimators import RegressorExtension_1x3
from sdu_estimators._sdu_estimators import RegressorExtension_3x6

from sdu_estimators._sdu_estimators import DREM_1x1
from sdu_estimators._sdu_estimators import DREM_1x2
from sdu_estimators._sdu_estimators import DREM_1x3
from sdu_estimators._sdu_estimators import DREM_3x6

from sdu_estimators._sdu_estimators import IntegrationMethod

from sdu_estimators._sdu_estimators import Sphere_3
from sdu_estimators._sdu_estimators import GradientEstimatorSphere_1x3

# from sdu_estimators._sdu_estimators import Kreisselmeier
# from sdu_estimators._sdu_estimators import RegressorExtension

# Export the version given in project metadata
from importlib import metadata

__version__ = metadata.version(__package__)
del metadata

def DREM(dt, gamma, theta_init, regressor_extension, r = 1.):
    # assert gamma.size() == theta_init.size()
    if isinstance(theta_init, (np.ndarray, list)):
        DIM_P = len(theta_init)
    else:
        DIM_P = 1.

    match DIM_P:
        case 1:
            theta_init = np.asarray([theta_init])
            gamma = np.asarray([gamma])
            return DREM_1x1(dt, gamma, theta_init, regressor_extension, r)

        case 2:
            # print()
            return DREM_1x2(dt, gamma, theta_init, regressor_extension, r)

        case 3:
            return DREM_1x3(dt, gamma, theta_init, regressor_extension, r)

        case _:
            raise ValueError(f"Wrong p-dimension. {DIM_P} is not supported.")

def Gradient(dt, gamma, theta_init, r = 1.):
    # assert gamma.size() == theta_init.size()
    if isinstance(theta_init, (np.ndarray, list)):
        DIM_P = len(theta_init)
    else:
        DIM_P = 1.

    match DIM_P:
        case 1:
            theta_init = np.asarray([theta_init])
            gamma = np.asarray([gamma])
            return GradientEstimator_1x1(dt, gamma, theta_init, r)

        case 2:
            # print()
            return GradientEstimator_1x2(dt, gamma, theta_init, r)

        case 3:
            return GradientEstimator_1x3(dt, gamma, theta_init, r)

        case _:
            raise ValueError(f"Wrong p-dimension. {DIM_P} is not supported.")