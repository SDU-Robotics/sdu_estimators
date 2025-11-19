#!/usr/bin/env python3
__all__ = (
    "parameter_estimators",
    "regressor_extensions",
    "math",
    "disturbance_observers"
)

import numpy as np

from ._sdu_estimators import parameter_estimators
from ._sdu_estimators import regressor_extensions
from ._sdu_estimators import math
from ._sdu_estimators import disturbance_observers

# Export the version given in project metadata
from importlib import metadata

__version__ = metadata.version(__package__)
del metadata


def _DREM(dt, gamma, theta_init, regressor_extension, r = 1.):
    # assert gamma.size() == theta_init.size()
    dim_p = 1
    if isinstance(theta_init, (np.ndarray, list)):
        dim_p = len(theta_init)

    match dim_p:
        case 1:
            theta_init = np.asarray([theta_init])
            gamma = np.asarray([gamma])
            return parameter_estimators.DREM_1x1(dt, gamma, theta_init, regressor_extension, r)

        case 2:
            # print()
            return parameter_estimators.DREM_1x2(dt, gamma, theta_init, regressor_extension, r)

        case 3:
            return parameter_estimators.DREM_1x3(dt, gamma, theta_init, regressor_extension, r)

        case _:
            raise ValueError(f"Wrong p-dimension. {dim_p} is not supported.")


parameter_estimators.DREM = _DREM


def _Gradient(dt, gamma, theta_init, r = 1., method=parameter_estimators.utils.Euler):
    # assert gamma.size() == theta_init.size()
    dim_p = 1
    if isinstance(theta_init, (np.ndarray, list)):
        dim_p = len(theta_init)

    match dim_p:
        case 1:
            theta_init = np.asarray([theta_init])
            gamma = np.asarray([gamma])
            return parameter_estimators.GradientEstimator_1x1(dt, gamma, theta_init, r, method)

        case 2:
            print(theta_init)
            print(gamma)
            return parameter_estimators.GradientEstimator_1x2(dt, gamma, theta_init, r, method)

        case 3:
            return parameter_estimators.GradientEstimator_1x3(dt, gamma, theta_init, r, method)

        case _:
            raise ValueError(f"Wrong p-dimension. {dim_p} is not supported.")


parameter_estimators.Gradient = _Gradient