*************
API Reference
*************

This page contains the API documentation of ``sdu_estimators``. The following classes are available:

Parameter estimators:

* :ref:`Estimator <estimator-api>`
* :ref:`Gradient Based Estimator <estimator-gradient-api>`
* :ref:`Dynamic Regressor Extension and Mixing (DREM) <estimator-drem-api>`
* :ref:`Regressor Extension <regressor-extension-api>`

State estimators:

* first

.. _estimator-api:

Parameter Estimator
===================

.. doxygenclass:: sdu_estimators::parameter_estimators::ParameterEstimator
    :project: sdu_estimators
    :members:


.. _estimator-gradient-api:

Gradient Estimator
==================

.. doxygenclass:: sdu_estimators::parameter_estimators::GradientEstimator
    :project: sdu_estimators
    :members:


.. _estimator-DREM-api:

Dynamic Regressor Extension and Mixing (DREM)
=============================================

.. doxygenclass:: sdu_estimators::parameter_estimators::DREM
    :project: sdu_estimators
    :members:


.. _regressor-extension-api:

Regressor Extension
===================

.. doxygenclass:: sdu_estimators::regressor_extensions::RegressorExtension
    :project: sdu_estimators
    :members:
