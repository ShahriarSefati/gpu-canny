cmake_minimum_required(VERSION 3.0)

set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA)
find_package(OPENCV)

include_directories("${CUDA_INCLUDE_DIRS}")
include_directories("${OpenCV_INCLUDE_DIRS}")

file( GLOB header *.hpp *.h )
file( GLOB cu  *.cu)
file( GLOB source *.cpp)

CUDA_ADD_EXECUTABLE(canny ${source} ${header} ${cu})
target_link_libraries(canny ${OpenCV_LIBS} ${CUDA_LIBS} )
