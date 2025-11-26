cmake_minimum_required(VERSION 3.16)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set a name and a version number for your project:
project(
  sdu_estimators
  VERSION 0.0.3
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
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# compile the library
add_subdirectory(src)

# compile the examples
add_subdirectory(examples)

# compile the tests
include(CTest)
if(BUILD_TESTING)
  if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ext/Catch2/CMakeLists.txt)
    message(
      FATAL_ERROR
        "Catch2 submodule not found. Please run 'git submodule update --init --recursive'."
    )
  endif()

  add_subdirectory(ext/Catch2)
  include(./ext/Catch2/contrib/Catch.cmake)
  add_subdirectory(tests)
endif()

if(BUILD_DOCS)
  # Add the documentation
  add_subdirectory(doc)
endif()

if(BUILD_PYTHON)
  if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ext/nanobind/CMakeLists.txt)
    message(
      FATAL_ERROR
        "nanobind submodule not found. Please run 'git submodule update --init --recursive'."
    )
  endif()

  if(CMAKE_VERSION VERSION_LESS 3.18)
    set(DEV_MODULE Development)
  else()
    set(DEV_MODULE Development.Module)
  endif()

  find_package(
    Python 3.8
    COMPONENTS Interpreter ${DEV_MODULE}
    REQUIRED)

  # Add Python bindings
  add_subdirectory(ext/nanobind)

  # Compile the nanobind module
  # nanobind_add_module(_sdu_estimators python/sdu_estimators/_sdu_estimators.cpp)
  nanobind_add_module(_sdu_estimators 
                      python/sdu_estimators/_sdu_estimators.cpp
                      python/sdu_estimators/_parameter_estimators.cpp
                      python/sdu_estimators/_regressor_extensions.cpp
                      python/sdu_estimators/_riemannian_manifolds.cpp
                      python/sdu_estimators/_disturbance_observers.cpp
                      python/sdu_estimators/_integrator.cpp)

  target_link_libraries(_sdu_estimators PUBLIC sdu_estimators)

  install(TARGETS _sdu_estimators DESTINATION ".")

  # nanobind_add_stub(
  #   sdu_estimators_stub
    
  #   MODULE sdu_estimators

  #   PYTHON_PATH $<TARGET_FILE_DIR:sdu_estimators>
  #   DEPENDS _sdu_estimators 
    
  #   RECURSIVE
  #   VERBOSE

  #   MARKER_FILE py.typed

  #   OUTPUT_PATH _sdu_estimators
  #   OUTPUT 
  #     _sdu_estimators/__init__pyi
  #   )

  #   install(TARGETS _sdu_estimators DESTINATION ".")
  #   install(FILES py.typed _sdu_estimators/__init__.pyi DESTINATION ".")
  #   # install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sdu_estimators.pyi
  #   #         ${CMAKE_CURRENT_BINARY_DIR}/py.typed DESTINATION .)
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
