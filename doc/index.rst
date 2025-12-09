
sdu_estimators
==============

|License badge| |Build badge| |Docs badge|

.. |License badge| image:: https://img.shields.io/badge/License-MIT-yellow.svg
  :target: https://opensource.org/licenses/MIT

.. |Build badge| image:: https://img.shields.io/github/actions/workflow/status/SDU-Robotics/sdu_estimators/ci.yml?branch=main
  :target: https://github.com/SDU-Robotics/sdu_estimators/actions/workflows/ci.yml

.. |Docs badge| image:: https://readthedocs.org/projects/sdu_estimators/badge/
  :target: https://sdu-estimators.readthedocs.io/

sdu_estimators is a C++ library containing online parameter estimation methods for general as well as specific applications developed at University of Southern Denmark (SDU).
The library is developed and maintained by Emil Lykke Diget of the `SDU Robotics <https://www.sdu.dk/en/forskning/sdurobotics>`_ group at University of Southern Denmark (SDU).
Python bindings are supplied for most of the functionality such that it can be used as a Python library.

---------------------
In this documentation
---------------------

.. grid:: 1 1 2 2
  
  .. grid-item:: :doc:`Tutorial <pages/tutorial/index>`

    **Start here**: A hands-on introduction to sdu_estimators for new users.

  .. grid-item:: :doc:`How-to guides <pages/how_to_guides/index>`

    **Step-by-step guides**: Covering common tasks and key functionalities.

.. grid:: 1 1 2 2
  :reverse:

  .. grid-item:: :doc:`Reference <pages/api/api>` 

    **Technical Information** - specification, APIs, architecture.

  .. grid-item:: :doc:`Explanation <pages/explanation/index>`

    **Discussion and clarification** of key topics.

---------------------

.. toctree::
  :hidden:
  :maxdepth: 3
  :caption: Table of Contents

  pages/tutorial/index
  pages/how_to_guides/index
  pages/api/api
  pages/explanation/index
  pages/bibliography