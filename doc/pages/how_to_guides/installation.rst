
.. _installation:

************
Installation
************

.. _building_sdu_estimators:

Building sdu_estimators
=======================

Prerequisites
-------------

Building sdu_estimators requires the following software installed:

* A C++17-compliant compiler.
* CMake `>= 3.9`.
* Eigen3 `>= 3.3` for linear algebra.
* Doxygen (optional, documentation building is skipped if missing)
* Python `>= 3.8` for building Python bindings

On debian-based linux distributions like Ubuntu, you can install the dependencies with:

.. code-block:: bash

    sudo apt install build-essential cmake python3-dev python3-pip libeigen3-dev

The following sequence of commands builds sdu_controllers.
It assumes that your current working directory is the top-level directory
of the freshly cloned repository:

.. code-block:: bash

   git submodule update --init --recursive
   mkdir build
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   make
   sudo make install

The build process can be customized with the following CMake variables,
which can be set by adding `-D<var>={ON, OFF}` to the `cmake` call:

* `BUILD_TESTING`: Enable building of the test suite (default: `ON`)
* `BUILD_DOCS`: Enable building the documentation (default: `ON`)
* `BUILD_PYTHON`: Enable building the Python bindings (default: `ON`)
* `BUILD_EXAMPLES`: Enable building the examples (default: `ON`)

Add the following to the CMakeLists.txt in your project to link to it:

.. code-block:: bash

    find_package(sdu_estimators REQUIRED)
    
    target_link_libraries(my_ext
        PRIVATE
        sdu_estimators::sdu-estimators
    )

If you wish to build and install the project as a Python project without
having access to C++ build artifacts like libraries and executables, you
can do so using `pip` from the root directory:

.. code-block:: bash

   git submodule update --init --recursive
   python3 -m pip install .