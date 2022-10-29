cmake_minimum_required(VERSION 3.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 14)

project(DBoW_Exercise1)

find_package(OpenCV REQUIRED)
message(STATUS ${OpenCV_INCLUDE_DIRS})
include_directories(src ${OpenCV_INCLUDE_DIRS})
message(STATUS ${OpenCV_LIBS})
link_libraries(${OpenCV_LIBS})

find_package(DBoW3 REQUIRED)
message(STATUS ${DBoW3_INCLUDE_DIRS})
message(STATUS ${DBoW3_LIBS})
include_directories(${DBoW3_INCLUDE_DIRS})
link_libraries(${DBoW3_LIBS})

find_package(Boost REQUIRED COMPONENTS filesystem system)
message(${Boost_INCLUDE_DIRS})
message(${Boost_LIBRARIES})



add_executable(create_voc_dir create_voc_dir.cpp)
target_include_directories(create_voc_dir PRIVATE ${BOOST_INCLUDE_DIRS})
target_link_libraries(create_voc_dir PRIVATE boost_system boost_filesystem)
