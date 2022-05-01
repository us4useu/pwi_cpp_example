cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(cpp_example LANGUAGES CXX CUDA)

# Conan generates FindXYZ.cmake files of dependencies into build directory --
# -- add it to CMAKE_MODULE_PATH, so we can use e.g. OpenCV package.
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_BINARY_DIR}")
find_package(OpenCV REQUIRED)

set(GPU_SRC
    # Imaging
#    imaging/kernels/Decimation.cuh
#    imaging/kernels/EnvelopeDetection.cuh
#    imaging/kernels/FirFilterSingleton.cuh
#    imaging/kernels/Kernel.cuh
#    imaging/kernels/KernelConstructionContext.h
#    imaging/kernels/KernelInitResult.h
#    imaging/kernels/LpFilterSingleton.cuh
#    imaging/kernels/QuadratureDemodulation.cuh
#    imaging/kernels/ReconstructLriPwi.cuh
#    imaging/kernels/RemapToLogicalOrder.cuh
#    imaging/kernels/Sum.h
#    imaging/kernels/ToBmode.cuh
#    imaging/kernels/Transpose.cuh
    imaging/CudaUtils.cuh
    imaging/kernels/RemapToLogicalOrder.cu
)

add_library(pwi_example_gpu ${GPU_SRC})
set_target_properties(pwi_example_gpu PROPERTIES CUDA_ARCHITECTURES 52)


# Main executable file.
# ARRUS package requires a compiler with C++17 support at least.
# Now, all the components should be C++17 at least.
set(CMAKE_CXX_STANDARD 17)
set(MAIN_SRC
    main.cpp
    gui.h
    pwi.h
    pwi.cpp
    menu.h
    menu.cpp
    common.h
    common.cpp

    logging/MyCustomLoggerFactory.h
    logging/MyCustomLogger.h

    imaging/NdArray.h
    imaging/DataType.h

    imaging/Kernel.h
    imaging/KernelConstructionContext.h
    imaging/KernelExecutionContext.h
    imaging/Metadata.h

    imaging/Pipeline.h
    imaging/Pipeline.cpp
    imaging/kernels/RemapToLogicalOrder.cpp
)

# TODO change the below path appropriately.
set(ARRUS_INCLUDE_PATH "/opt/arrus/develop/include")
# TODO change the below path appropriately.
set(ARRUS_LIB_PATH "/opt/arrus/develop/lib64")

add_executable(pwi_example ${MAIN_SRC})
target_link_directories(pwi_example PRIVATE ${ARRUS_LIB_PATH})
target_include_directories(pwi_example PRIVATE ${ARRUS_INCLUDE_PATH})
target_link_libraries(pwi_example PRIVATE pwi_example_gpu opencv::opencv)


