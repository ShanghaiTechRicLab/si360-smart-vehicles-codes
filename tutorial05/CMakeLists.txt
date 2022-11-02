cmake_minimum_required(VERSION 3.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 14)
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache) # Less useful to do it for linking, see edit2
endif(CCACHE_FOUND)

project(Tutorial05_LiDARLoopClosure)

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


