<p>
</p>
<div align="center">
    <div>
    <img width=25% src="doc/_static/sdu_estimators_logo.png" style="vertical-align:top;margin-bottom:-20px">
    <h1>sdu_estimators</h1>
    <p>A C++ library containing estimators developed at University of Southern Denmark (SDU).</p>
    </div>
</div>
<div align="center">
<p>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/SDU-Robotics/sdu_estimators/ci.yml?branch=main)](https://github.com/SDU-Robotics/sdu_estimators/actions/workflows/ci.yml)
[![PyPI Release](https://img.shields.io/pypi/v/sdu_estimators.svg)](https://pypi.org/project/sdu_estimators)
[![Documentation Status](https://readthedocs.org/projects/sdu_estimators/badge/)](https://sdu-estimators.readthedocs.io/)
<!-- [![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=SDU-Robotics_sdu_estimators&metric=alert_status)](https://sonarcloud.io/dashboard?id=SDU-Robotics_sdu_estimators) !-->
</div>

# Prerequisites
Building sdu_estimators requires the following software installed:

* A C++17-compliant compiler
* CMake `>= 3.9`
* Eigen3 `>= 3.3` for linear algebra.
* Doxygen (optional, documentation building is skipped if missing)
* Python `>= 3.8` for building Python bindings

# Building sdu_estimators

The following sequence of commands builds sdu_estimators.
It assumes that your current working directory is the top-level directory
of the freshly cloned repository:

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

The build process can be customized with the following CMake variables,
which can be set by adding `-D<var>={ON, OFF}` to the `cmake` call:

* `BUILD_TESTING`: Enable building of the test suite (default: `ON`)
* `BUILD_DOCS`: Enable building the documentation (default: `ON`)
* `BUILD_PYTHON`: Enable building the Python bindings (default: `ON`)


If you wish to build and install the project as a Python project without
having access to C++ build artifacts like libraries and executables, you
can do so using `pip` from the root directory:

```
python -m pip install .
```

# Testing sdu_estimators

When built according to the above explanation (with `-DBUILD_TESTING=ON`),
the C++ test suite of `sdu_estimators` can be run using
`ctest` from the build directory:

```
cd build
ctest
```

The Python test suite can be run by first `pip`-installing the Python package
and then running `pytest` from the top-level directory:

```
python -m pip install .
pytest
```

# Documentation

sdu_estimators provides a Sphinx-based documentation, that can
be browsed [online at readthedocs.org](https://sdu-estimators.readthedocs.io).
To build it locally, first ensure the requirements are installed by running this command from the top-level source directory:

```
pip install -r doc/requirements.txt
```

Then build the sphinx documentation from the top-level build directory:

```
cmake --build . --target sphinx-doc
```

The web documentation can then be browsed by opening `doc/sphinx/index.html` in your browser.
