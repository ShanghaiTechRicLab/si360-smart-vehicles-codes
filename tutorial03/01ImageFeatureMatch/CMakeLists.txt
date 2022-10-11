cmake_minimum_required(VERSION 3.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(ImageFeaturePointExtractor)

find_package(OpenCV REQUIRED)

message(STATUS ${OpenCV_INCLUDE_DIRS})

include_directories(src ${OpenCV_INCLUDE_DIRS})

add_executable(image_feature_extractor src/extractor.cpp)
target_link_libraries(image_feature_extractor PRIVATE ${OpenCV_LIBS})

add_library(feature_extractor_lib SHARED src/feature_extractor.cpp)
target_link_libraries(feature_extractor_lib PUBLIC ${OpenCV_LIBS})

add_executable(test_extractor_lib src/test_extractor.cpp)
target_link_libraries(test_extractor_lib PRIVATE feature_extractor_lib)

add_executable(image_feature_matcher src/matcher.cpp)
target_link_libraries(image_feature_matcher PRIVATE feature_extractor_lib)