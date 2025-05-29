cmake_minimum_required(VERSION 3.10)
project(TinyTorch)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_CUDA "Use CUDA" ON)
option(USE_BLAS "Use BLAS" ON)
option(USE_OPENCV "Use OPENCV" ON)

if (APPLE)
    set(USE_CUDA OFF)
endif ()

message(STATUS "USE_CUDA ${USE_CUDA}")
message(STATUS "USE_BLAS ${USE_BLAS}")
message(STATUS "USE_OPENCV ${USE_OPENCV}")

if (${USE_CUDA})
    add_definitions(-DUSE_CUDA)
endif ()


if (${USE_OPENCV})
    # you need set environment path called OpenCV_DIR, usually path/opencv/build
    if(NOT DEFINED ENV{OpenCV_DIR})
        message(FATAL_ERROR "OpenCV_DIR environment variable is not set. Please set it to the path of your OpenCV build directory.")
    endif()
    find_package(OpenCV REQUIRED COMPONENTS core imgcodecs)
    set(OpenCV_STATIC ON)
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

    add_definitions(-DUSE_OPENCV)
endif()


include(FetchContent)
FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.12.0 # 或者任何其他版本标签
)
# 确保库被获取并使其准备好被使用
FetchContent_MakeAvailable(json)


# 设置CUDA的编译标志
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --keep --keep-dir tmp -gencode arch=compute_50,code=sm_50 -v")
add_subdirectory(TinyTorch)
add_subdirectory(demo)
option(BUILD_TEST "Whether or not to build the tests" ON)
if (${BUILD_TEST})
    message(STATUS "Building tests")

    option(TEST_CUDA "Test CUDA" OFF)
    message(STATUS "TEST_CUDA ${TEST_CUDA}")

    if (${TEST_CUDA})
        add_definitions(-DDEFAULT_DEVICE_CUDA)
    endif ()

    enable_testing()
    add_subdirectory(test)
endif ()
