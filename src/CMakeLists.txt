add_library(sdu_estimators sdu_estimators.cpp)
target_link_libraries(sdu_estimators Eigen3::Eigen)
target_include_directories(
  sdu_estimators PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
                        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
