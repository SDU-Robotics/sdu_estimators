macro(add_example example_name folder)
    if(folder STREQUAL "")
        set(folder ".")
    endif()
    add_executable(${example_name} ${folder}/${example_name}.cpp)
    target_link_libraries(${example_name} PRIVATE sdu_estimators)
endmacro()

if(MSVC)
  add_compile_options(/bigobj)
endif()


add_example(example_gradient_estimator "parameter_estimators")
add_example(example_DREM_estimator "parameter_estimators")
add_example(example_SS "state_estimators")
add_example(example_luenberger "state_estimators")
add_example(example_UIO "disturbance_observers")
add_example(example_momentum_observer "disturbance_observers")

# increase stack size
add_example(example_beam "structural_parameters")
target_compile_definitions(example_beam PRIVATE EIGEN_STACK_ALLOCATION_LIMIT=0)

add_example(example_utils "state_estimators")

# manifolds
add_example(example_gradient_estimator_sphere "manifold")
add_example(test_manifold_init "manifold")