cmake_minimum_required(VERSION 3.10)

# Set project name
project(ExtendedKalmanFilter)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add Eigen as a dependency
# If Eigen is installed globally, find it
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Include directories for source files
include_directories(${CMAKE_SOURCE_DIR}/src)

# Add the executable
add_executable(extended_kalman_filter src/main.cpp src/extended_kalman_filter/extended_kalman_filter.hpp)

# Link Eigen to the target (although Eigen is header-only, you still need to link it to your project)
target_link_libraries(extended_kalman_filter Eigen3::Eigen)
