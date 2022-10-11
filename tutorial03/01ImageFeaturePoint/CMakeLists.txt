cmake_minimum_required(VERSION 3.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(ImageFeaturePointExtractor)


find_package(OpenCV REQUIRED)

add_executable(image_feature_point_extractor src/extractor.cpp)
target_link_libraries(image_feature_point_extractor  PRIVATE ${OpenCV_LIBS})
target_include_directories(image_feature_point_extractor PRIVATE src ${OpenCV_INCLUDE_DIRS})
