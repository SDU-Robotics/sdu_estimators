add_library(sdu_estimators
    sdu_estimators.cpp
    estimators/gradient_estimator.cpp
    estimators/drem.cpp
    ../include/sdu_estimators/sdu_estimators.hpp
    ../include/sdu_estimators/estimators/estimator.hpp
    ../include/sdu_estimators/estimators/gradient_estimator.hpp
    ../include/sdu_estimators/estimators/drem.hpp

    regressor_extensions/kreisselmeier.cpp
    ../include/sdu_estimators/regressor_extensions/kreisselmeier.hpp
)

target_link_libraries(sdu_estimators Eigen3::Eigen)

target_include_directories(
  sdu_estimators PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
                        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
