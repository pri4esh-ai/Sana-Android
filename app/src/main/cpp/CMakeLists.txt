cmake_minimum_required(VERSION 3.22.1)

project(sana_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MNN source is cloned by GitHub Actions.
set(MNN_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/MNN/source"
)

# Prebuilt MNN library copied by the GitHub Actions workflow.
set(MNN_LIB_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}"
)

set(MNN_LIB
    "${MNN_LIB_DIR}/libMNN.so"
)

if(NOT EXISTS "${MNN_LIB}")
    message(FATAL_ERROR
        "MNN library not found: ${MNN_LIB}"
    )
endif()

if(NOT EXISTS "${MNN_SOURCE_DIR}")
    message(FATAL_ERROR
        "MNN source not found: ${MNN_SOURCE_DIR}"
    )
endif()

message(STATUS "======================================")
message(STATUS "SANA NATIVE BUILD")
message(STATUS "======================================")
message(STATUS "ABI: ${ANDROID_ABI}")
message(STATUS "MNN source: ${MNN_SOURCE_DIR}")
message(STATUS "MNN library: ${MNN_LIB}")
message(STATUS "======================================")

add_library(MNN SHARED IMPORTED)

set_target_properties(
    MNN
    PROPERTIES
    IMPORTED_LOCATION "${MNN_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${MNN_SOURCE_DIR}"
)

add_library(
    sana_native
    SHARED
    sana_jni.cpp
)

target_include_directories(
    sana_native
    PRIVATE
    "${MNN_SOURCE_DIR}"
)

find_library(log-lib log)
find_library(android-lib android)

target_link_libraries(
    sana_native
    MNN
    ${log-lib}
    ${android-lib}
)
