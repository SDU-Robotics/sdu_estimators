# add_library(sdu_estimators INTERFACE
#         #    estimators/gradient_estimator.cpp
# #    estimators/drem.cpp

#     ../include/sdu_estimators/parameter_estimators/parameter_estimator.hpp
#     ../include/sdu_estimators/parameter_estimators/gradient_estimator.hpp
#     ../include/sdu_estimators/parameter_estimators/drem.hpp
#     ../include/sdu_estimators/regressor_extensions/kreisselmeier.hpp
#     ../include/sdu_estimators/regressor_extensions/lti.hpp
# #    ../include/sdu_estimators/regressor_extensions/delay.hpp

#     ../include/sdu_estimators/state_estimators/state_space_model.hpp
#     ../include/sdu_estimators/state_estimators/utils.hpp
# #    ../include/sdu_estimators/state_estimators/greet_fortran.f

# #    state_estimators/state_space_model.cpp
#     ../include/sdu_estimators/state_estimators/luenberger_observer.hpp
#     ../include/sdu_estimators/state_estimators/unknown_input_observer.hpp
# #    state_estimators/luenberger_observer.cpp
# #    state_estimators/unknown_input_observer.cpp
# )

add_library(sdu_estimators INTERFACE)

target_link_libraries(sdu_estimators INTERFACE
        Eigen3::Eigen)

#target_include_directories(
#  sdu_estimators PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
#                        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(
  sdu_estimators INTERFACE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
