#!/usr/bin/env python3

from ._sdu_estimators import parameter_estimators
from ._sdu_estimators import regressor_extensions
from ._sdu_estimators import math
from ._sdu_estimators import disturbance_observers
from ._sdu_estimators import integrator

from importlib import metadata

__version__ = metadata.version(__package__)

del metadata

import numpy as np


def _DREM(dt, gamma, theta_init, regressor_extension, r = 1., method=integrator.RK4):
    # assert gamma.size() == theta_init.size()
    dim_p = 1
    if isinstance(theta_init, (np.ndarray, list)):
        dim_p = len(theta_init)

    match dim_p:
        case 1:
            theta_init = np.asarray([theta_init])
            gamma = np.asarray([gamma])
            return parameter_estimators.DREM_1x1(dt, gamma, theta_init, regressor_extension, r, method)

        case 2:
            # print()
            return parameter_estimators.DREM_1x2(dt, gamma, theta_init, regressor_extension, r, method)

        case 3:
            return parameter_estimators.DREM_1x3(dt, gamma, theta_init, regressor_extension, r, method)

        case _:
            raise ValueError(f"Wrong p-dimension. {dim_p} is not supported.")


parameter_estimators.DREM = _DREM


def _CascadedDREM(dt, a, DIM_N, DIM_P, method=integrator.RK4):
    match DIM_N:
        case 4:
            match DIM_P:
                case 2:
                    return parameter_estimators.CascadedDREM_4x2(dt, a, method)


parameter_estimators.CascadedDREM = _CascadedDREM


def _Gradient(dt, gamma, theta_init, r = 1., method=integrator.RK4):
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