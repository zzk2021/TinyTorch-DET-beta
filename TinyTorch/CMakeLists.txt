cmake_minimum_required(VERSION 3.10)
project(TinyTorch)


if (${USE_CUDA})
    set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;89")
    find_package(CUDA REQUIRED)
    enable_language(CUDA)

    file(GLOB_RECURSE TinyTorch_src_cu "./TensorImpl/*_cuda.cu")

    if (${USE_CUDNN})
        if(NOT DEFINED ENV{cuDNN_DIR})
            message(FATAL_ERROR "cuDNN_DIR environment variable is not set. Please set it to the path of your CUDNN build directory.")
        endif()
        set(CUDNN_DIR "$ENV{cuDNN_DIR}")
        set(CMAKE_MODULE_PATH "${CUDNN_DIR}/cudnn_samples/cmake")
        set(cuDNN_INCLUDE_DIR "${CUDNN_DIR}/include/12.8")
        set(cuDNN_LIBRARY_DIR "${CUDNN_DIR}/lib/12.8/x64")
        find_package(cuDNN REQUIRED)
        file(GLOB_RECURSE TinyTorch_src_cudnn "./TensorImpl/*_cudnn.cu")
    else ()
        set(TinyTorch_src_cudnn "")
    endif()
else ()
    set(TinyTorch_src_cu "")
endif ()

file(GLOB_RECURSE TinyTorch_src "./*.cpp" "./TensorImpl/*.cpp")
add_library(${PROJECT_NAME} ${TinyTorch_src} ${TinyTorch_src_cu} ${TinyTorch_src_cudnn})

if (CMAKE_BUILD_TYPE STREQUAL Debug)
    add_definitions(-DDEBUG)
elseif (CMAKE_BUILD_TYPE STREQUAL Release)
    if (MSVC)
        set_source_files_properties(${TinyTorch_src} PROPERTIES COMPILE_FLAGS "/Ox")
    else ()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    endif ()
endif ()

if (${USE_CUDA})
    include_directories(${CUDA_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_cublas_LIBRARY})
endif ()

if (${USE_CUDNN})
    include_directories(${cuDNN_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} ${cuDNN_LIBRARY_DIR})
endif ()

if (${USE_OPENCV})
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${OpenCV_3RDPARTY_LIBS} ${OpenCV_LIBS})
endif ()

if (${USE_BLAS})
    add_definitions(-DUSE_BLAS)
    if (MSVC)
        target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/OpenBLAS/win64-64/include)
        target_link_libraries(${PROJECT_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/OpenBLAS/win64-64/lib/libopenblas.dll.a")
    elseif (APPLE)
        find_library(ACCELERATE_LIBRARY Accelerate)
        target_link_libraries(${PROJECT_NAME} ${ACCELERATE_LIBRARY})
    elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
        target_include_directories(${PROJECT_NAME} PRIVATE "/usr/include/openblas")
        target_link_libraries(${PROJECT_NAME} openblas)
    endif ()
endif ()


if (${USE_DEBUG})
    if(UNIX)
        target_link_libraries(${PROJECT_NAME} backtrace)
        target_compile_definitions(${PROJECT_NAME} PRIVATE USE_BACKTRACE)
    elseif(WIN32)
        target_link_libraries(${PROJECT_NAME} DbgHelp)
        target_compile_definitions(${PROJECT_NAME} PRIVATE USE_DBGHELP)
    endif()
endif()
