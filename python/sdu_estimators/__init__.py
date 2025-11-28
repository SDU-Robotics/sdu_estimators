#!/usr/bin/env python3

__all__ = (
    "parameter_estimators",
    "regressor_extensions",
    "math",
    "disturbance_observers",
    "integrator"
)

from ._sdu_estimators import __doc__
from ._sdu_estimators import parameter_estimators as parameter_estimators
from ._sdu_estimators import regressor_extensions as regressor_extensions
from ._sdu_estimators import math as math
from ._sdu_estimators import disturbance_observers as disturbance_observers
from ._sdu_estimators import integrator as integrator


# from . import (
#     parameter_estimators as parameter_estimators,
#     regressor_extensions as regressor_extensions,
#     math as math,
#     disturbance_observerser as disturbance_observerser,
#     integrator as integrator
# )

from importlib import metadata

__version__ = metadata.version(__package__)

del metadata

import numpy as np


class _ParameterEstimator:
    """ Parent class for parameter estimators """
    def __init__(self, solver):
        self.solver = solver

    def step(self, y : np.ndarray, phi : np.ndarray):
        self.solver.step(y, phi)

    def get_estimate(self):
        return self.solver.get_estimate()


parameter_estimators.ParameterEstimator = _ParameterEstimator
parameter_estimators.ParameterEstimator.__name__ = "ParameterEstimator"
parameter_estimators.ParameterEstimator.__qualname__ = "ParameterEstimator"  


class _DREM(_ParameterEstimator):
    def __init__(self, dt, gamma, theta_init, regressor_extension, r = 1., method=integrator.IntegrationMethod.RK4):
        # assert gamma.size() == theta_init.size()
        dim_p = 1
        if isinstance(theta_init, (np.ndarray, list)):
            dim_p = len(theta_init)

        match dim_p:
            case 1:
                theta_init = np.asarray([theta_init])
                gamma = np.asarray([gamma])
                solver = parameter_estimators.DREM_1x1(dt, gamma, theta_init, regressor_extension, r, method)

            case 2:
                # print()
                solver = parameter_estimators.DREM_1x2(dt, gamma, theta_init, regressor_extension, r, method)

            case 3:
                solver = parameter_estimators.DREM_1x3(dt, gamma, theta_init, regressor_extension, r, method)

            case _:
                raise ValueError(f"Wrong p-dimension. {dim_p} is not supported.")

        super().__init__(solver)


parameter_estimators.DREM = _DREM


class _CascadedDREM(_ParameterEstimator):
    def __init__(self, dt, a, DIM_N, DIM_P, method=integrator.IntegrationMethod.RK4):
        match DIM_N:
            case 4:
                match DIM_P:
                    case 2:
                        solver = parameter_estimators.CascadedDREM_4x2(dt, a, method)

                    case _:
                        raise ValueError(f"Wrong P-dimension. {DIM_P} is not supported.")

            case _:
                raise ValueError(f"Wrong N-dimension. {DIM_N} is not supported.")

        super().__init__(solver)

        
parameter_estimators.CascadedDREM = _CascadedDREM
parameter_estimators.CascadedDREM.__name__ = "CascadedDREM"
parameter_estimators.CascadedDREM.__qualname__ = "CascadedDREM"


class _GradientEstimator(_ParameterEstimator):
    """ 
    Instantiate the correct Gradient Estimator depending on the size of the theta input.
    """
    def __init__(self, dt : float, gamma : np.ndarray, theta_init : np.ndarray, r = 1., method=integrator.IntegrationMethod.RK4):
        dim_p = 1
        if isinstance(theta_init, (np.ndarray, list)):
            dim_p = len(theta_init)

        match dim_p:
            case 1:
                theta_init = np.asarray([theta_init])
                gamma = np.asarray([gamma])
                solver = parameter_estimators.GradientEstimator_1x1(dt, gamma, theta_init, r, method)

            case 2:
                print(theta_init)
                print(gamma)
                solver = parameter_estimators.GradientEstimator_1x2(dt, gamma, theta_init, r, method)

            case 3:
                solver = parameter_estimators.GradientEstimator_1x3(dt, gamma, theta_init, r, method)

            case _:
                raise ValueError(f"Wrong p-dimension. {dim_p} is not supported.")    
        
        super().__init__(solver)
        

parameter_estimators.GradientEstimator = _GradientEstimator
parameter_estimators.GradientEstimator.__name__ = "GradientEstimator"
parameter_estimators.GradientEstimator.__qualname__ = "GradientEstimator"


# def _Gradient(dt : float, gamma : np.ndarray, theta_init : np.ndarray, r = 1., method=integrator.IntegrationMethod.RK4):
#     # assert gamma.size() == theta_init.size()
#     dim_p = 1
#     if isinstance(theta_init, (np.ndarray, list)):
#         dim_p = len(theta_init)

#     match dim_p:
#         case 1:
#             theta_init = np.asarray([theta_init])
#             gamma = np.asarray([gamma])
#             return parameter_estimators.GradientEstimator_1x1(dt, gamma, theta_init, r, method)

#         case 2:
#             print(theta_init)
#             print(gamma)
#             return parameter_estimators.GradientEstimator_1x2(dt, gamma, theta_init, r, method)

#         case 3:
#             return parameter_estimators.GradientEstimator_1x3(dt, gamma, theta_init, r, method)

#         case _:
#             raise ValueError(f"Wrong p-dimension. {dim_p} is not supported.")


# parameter_estimators.Gradient = _Gradient