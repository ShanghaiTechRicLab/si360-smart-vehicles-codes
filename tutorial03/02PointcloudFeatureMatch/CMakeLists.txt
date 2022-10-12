cmake_minimum_required(VERSION 3.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(PointcloudFeatureMatch)

find_package(
    PCL
    COMPONENTS
        io
        search
        keypoints
        features
        range_image
        common
        console
        visualization
    REQUIRED
)

message(STATUS "pcl include dirs:${PCL_INCLUDE_DIRS}")

message(STATUS "pcl libraries: ${PCL_LIBRARIES}")

include_directories(src ${PCL_INCLUDE_DIRS})

add_executable(descriptors src/descriptors.cpp)
target_link_libraries(descriptors PRIVATE ${PCL_LIBRARIES})

