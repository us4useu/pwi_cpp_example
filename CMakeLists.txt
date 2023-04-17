cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(cpp_example LANGUAGES CXX CUDA)

# Conan generates FindXYZ.cmake files of dependencies into build directory --
# -- add it to CMAKE_MODULE_PATH, so we can use e.g. OpenCV package.
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_BINARY_DIR}")
find_package(OpenCV REQUIRED)
# TODO change the below path appropriately.
set(ARRUS_INCLUDE_PATH "/home/pjarosik/src/arrus/install/include")
# TODO change the below path appropriately.
set(ARRUS_LIB_PATH "/home/pjarosik/src/arrus/install/lib64")

set(GPU_SRC
    # Imaging
    imaging/CudaUtils.cuh
    imaging/kernels/RemapToLogicalOrder.cu
    imaging/kernels/Transpose.cu
    imaging/kernels/BandpassFilter.cu
    imaging/kernels/DigitalDownConversion.cu
    imaging/kernels/ReconstructHri.cu
    imaging/kernels/EnvelopeDetection.cu
    imaging/kernels/ToBmode.cu
    imaging/kernels/RemapToLogicalOrder2.cu
    imaging/kernels/ToComplex.cu
    imaging/kernels/Real.cu
    imaging/kernels/Imag.cu
    imaging/kernels/Phase.cu
)

add_library(pwi_example_gpu ${GPU_SRC})
target_include_directories(pwi_example_gpu PRIVATE ${CMAKE_SOURCE_DIR})
set_target_properties(pwi_example_gpu PROPERTIES CUDA_ARCHITECTURES 52)

# Main executable file.
# ARRUS package requires a compiler with C++17 support at least.
# Now, all the components should be C++17 at least.
set(CMAKE_CXX_STANDARD 17)
set(MAIN_SRC
    gui.h
    imaging/pwi.h
    imaging/pwi.cpp
    menu.cpp
    common.h
    common.cpp
    imaging/utils.h

    logging/MyCustomLoggerFactory.h
    logging/MyCustomLogger.h

    imaging/NdArray.h
    imaging/DataType.h
    imaging/ProbeModelExt.h

    imaging/Operation.h
    imaging/Kernel.h
    imaging/KernelConstructionContext.h
    imaging/KernelExecutionContext.h
    imaging/Metadata.h

    imaging/PipelineRunner.h
    imaging/PipelineRunner.cpp

    imaging/ops/Pipeline.h

    imaging/ops/RemapToLogicalOrder.h
    imaging/ops/Transpose.h
    imaging/ops/BandpassFilter.h
    imaging/ops/DigitalDownConversion.h
    imaging/ops/ReconstructHri.h
    imaging/ops/EnvelopeDetection.h
    imaging/ops/ToBMode.h
    imaging/ops/RemapToLogicalOrder2.h
    imaging/ops/ToComplex.h
    imaging/ops/Real.h
    imaging/ops/Imag.h
    imaging/ops/Phase.h

    imaging/kernels/RemapToLogicalOrder.cpp
    imaging/kernels/Transpose.cpp
    imaging/kernels/BandpassFilter.cpp
    imaging/kernels/DigitalDownConversion.cpp
    imaging/kernels/ReconstructHri.cpp
    imaging/kernels/EnvelopeDetection.cpp
    imaging/kernels/ToBmode.cpp
    imaging/kernels/RemapToLogicalOrder2.cpp
    imaging/kernels/ToComplex.cpp
    imaging/kernels/Real.cpp
    imaging/kernels/Imag.cpp
    imaging/kernels/Phase.cpp

    imaging/kernels/RemapToLogicalOrder.h
    imaging/kernels/Transpose.h
    imaging/kernels/BandpassFilter.h
    imaging/kernels/DigitalDownConversion.h
    imaging/kernels/ReconstructHri.h
    imaging/kernels/EnvelopeDetection.h
    imaging/kernels/ToBmode.h
    imaging/kernels/RemapToLogicalOrder2.h
    imaging/kernels/ToComplex.h
    imaging/kernels/Real.h
    imaging/kernels/Imag.h
    imaging/kernels/Phase.h
)

# RF data example.
add_executable(rf_example rf_example.cpp ${MAIN_SRC})
target_include_directories(rf_example PRIVATE ${ARRUS_INCLUDE_PATH} ${CMAKE_SOURCE_DIR})
target_link_directories(rf_example PRIVATE ${ARRUS_LIB_PATH})
target_link_libraries(rf_example PRIVATE pwi_example_gpu opencv::opencv arrus-core)

# BMode data example.
add_executable(bmode_example bmode_example.cpp ${MAIN_SRC})
target_include_directories(bmode_example PRIVATE ${ARRUS_INCLUDE_PATH} ${CMAKE_SOURCE_DIR})
target_link_directories(bmode_example PRIVATE ${ARRUS_LIB_PATH})
target_link_libraries(bmode_example PRIVATE pwi_example_gpu opencv::opencv arrus-core)

# RF data example.
add_executable(ddc_example ddc_example.cpp ${MAIN_SRC})
target_include_directories(ddc_example PRIVATE ${ARRUS_INCLUDE_PATH} ${CMAKE_SOURCE_DIR})
target_link_directories(ddc_example PRIVATE ${ARRUS_LIB_PATH})
target_link_libraries(ddc_example PRIVATE pwi_example_gpu opencv::opencv arrus-core)

