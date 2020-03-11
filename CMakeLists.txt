cmake_minimum_required(VERSION 3.0.0)
project(risk_assess VERSION 0.1.0)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

include(CTest)
enable_testing()

add_library(test_cpp_moment_dynamics SHARED risk_assess/uncertain_agent/moment_dynamics.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)