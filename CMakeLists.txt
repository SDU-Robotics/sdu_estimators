cmake_minimum_required(VERSION 3.9)

# Set a name and a version number for your project:
project(
  sdu_estimators
  VERSION 0.0.1
  LANGUAGES CXX)

# Initialize some default paths
include(GNUInstallDirs)

# Define the minimum C++ standard that is required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compilation options
option(BUILD_PYTHON "Enable building of Python bindings" ON)
option(BUILD_DOCS "Enable building of documentation" ON)

# find dependencies
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

# compile the library
add_subdirectory(src)

# compile the examples
add_subdirectory(examples)

# compile the tests
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(ext/Catch2)
  include(./ext/Catch2/contrib/Catch.cmake)
  add_subdirectory(tests)
endif()

if(BUILD_DOCS)
  # Add the documentation
  add_subdirectory(doc)
endif()
if(BUILD_PYTHON)
  if (CMAKE_VERSION VERSION_LESS 3.18)
    set(DEV_MODULE Development)
  else()
    set(DEV_MODULE Development.Module)
  endif()

  find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

  # Add Python bindings
  add_subdirectory(ext/nanobind)

  # Compile the nanobind module
  nanobind_add_module(_sdu_estimators python/sdu_estimators/_sdu_estimators.cpp)
  target_link_libraries(_sdu_estimators PUBLIC sdu_estimators)

  nanobind_add_stub(
          _sdu_estimators
          INSTALL_TIME
          MODULE _sdu_estimators
          OUTPUT _sdu_estimators.pyi
          MARKER_FILE py.typed
          PYTHON_PATH "."

  # Install the Python module shared library
  install(TARGETS _sdu_estimators DESTINATION .)

  install(FILES py.typed my_ext.pyi DESTINATION ".")
  )

endif()

# Add an alias target for use if this project is included as a subproject in
# another project
add_library(sdu_estimators::sdu_estimators ALIAS sdu_estimators)

# Install targets and configuration
install(
  TARGETS sdu_estimators
  EXPORT sdu_estimators-config
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(
  EXPORT sdu_estimators-config
  NAMESPACE sdu_estimators::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdu_estimators)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# This prints a summary of found dependencies
include(FeatureSummary)
feature_summary(WHAT ALL)
