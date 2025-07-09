cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0146 NEW)

project(bilibili_vtuber LANGUAGES CXX)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

if(NOT CMAKE_SIZEOF_VOID_P)
set(CMAKE_SIZEOF_VOID_P 8)  # replace 8 with 4 for 32-bit machine
endif()

# OpenCV
set(OpenCV_DIR "C:/Program Files/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ONNX Runtime
set(ONNXRUNTIME_LIB_PATH "C:/Program Files/onnxruntime")
include_directories(${ONNXRUNTIME_LIB_PATH}/include)
link_directories(${ONNXRUNTIME_LIB_PATH}/lib)

# onnxModel - Create DLL that can run on systems without CUDA
add_library(onnxModel SHARED src/onnxModel.cpp src/onnxModel.def)

# Link with ONNX Runtime and OpenCV (no CUDA linking at build time)
target_link_libraries(onnxModel PRIVATE onnxruntime ${OpenCV_LIBS})

# Set runtime library to multi-threaded DLL
if(MSVC)
    set_property(TARGET onnxModel PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

# Copy required DLLs to output directory
add_custom_command(TARGET onnxModel POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_LIB_PATH}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:onnxModel>
)

# Copy ONNX Runtime providers shared DLL (required for CUDA support)
add_custom_command(TARGET onnxModel POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_LIB_PATH}/lib/onnxruntime_providers_shared.dll"
        $<TARGET_FILE_DIR:onnxModel>
)

# Copy ONNX Runtime CUDA providers DLL (required for CUDA inference)
add_custom_command(TARGET onnxModel POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_LIB_PATH}/lib/onnxruntime_providers_cuda.dll"
        $<TARGET_FILE_DIR:onnxModel>
)

# Copy OpenCV DLLs to output directory
add_custom_command(TARGET onnxModel POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OpenCV_DIR}/x64/vc16/bin/opencv_world4110.dll"
        $<TARGET_FILE_DIR:onnxModel>
)
