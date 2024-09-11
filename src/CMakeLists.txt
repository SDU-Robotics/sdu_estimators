add_library(sdu_estimators
    sdu_estimators.cpp
    estimators/gradient_estimator.cpp
    estimators/drem.cpp
    ../include/sdu_estimators/sdu_estimators.hpp
    ../include/sdu_estimators/parameter_estimators/parameter_estimator.hpp
    ../include/sdu_estimators/parameter_estimators/gradient_estimator.hpp
    ../include/sdu_estimators/parameter_estimators/drem.hpp

    regressor_extensions/kreisselmeier.cpp
    ../include/sdu_estimators/regressor_extensions/kreisselmeier.hpp

    ../include/sdu_estimators/state_estimators/state_estimator.hpp
    ../include/sdu_estimators/state_estimators/state_space_model.hpp
    ../include/sdu_estimators/state_estimators/utils.hpp
    state_estimators/state_space_model.cpp
)

target_link_libraries(sdu_estimators Eigen3::Eigen)

target_include_directories(
  sdu_estimators PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
                        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
