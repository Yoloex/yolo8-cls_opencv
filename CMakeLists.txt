# CMakeList.txt : CMake project for yolo8-cls, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("yolo8-cls")

# Add source to this project's executable.
add_executable (yolo8-cls "yolo8-cls.cpp" "yolo8-cls.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET yolo8-cls PROPERTY CXX_STANDARD 20)
endif()

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(yolo8-cls ${OpenCV_LIBS})