cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0146 NEW)

project(bilibili_vtuber LANGUAGES CXX CUDA)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

# OpenCV
set(OpenCV_DIR "C:/Program Files/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ONNX Runtime
set(ONNXRUNTIME_LIB_PATH "C:/Program Files/onnxruntime")
include_directories(${ONNXRUNTIME_LIB_PATH}/include)
link_directories(${ONNXRUNTIME_LIB_PATH}/lib)

# onnxModel
add_library(onnxModel SHARED src/onnxModel.cpp)
target_link_libraries(onnxModel PRIVATE cuda cudart onnxruntime ${OpenCV_LIBS})