cmake_minimum_required(VERSION 3.22.1)

project(sana_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "======================================")
message(STATUS "SANA NATIVE BUILD")
message(STATUS "======================================")
message(STATUS "ANDROID ABI: ${ANDROID_ABI}")

# Project root:
# app/src/main/cpp
#       -> app
#       -> main
#       -> src
#       -> Sana-Android
get_filename_component(
    PROJECT_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../"
    ABSOLUTE
)

set(MNN_ROOT
    "${PROJECT_ROOT}/third_party/MNN"
)

# IMPORTANT:
# MNN/Interpreter.hpp is under:
# third_party/MNN/include/MNN/Interpreter.hpp
set(MNN_INCLUDE_DIR
    "${MNN_ROOT}/include"
)

if(NOT EXISTS "${MNN_INCLUDE_DIR}/MNN/Interpreter.hpp")

    message(FATAL_ERROR
        "MNN/Interpreter.hpp NOT FOUND.\n"
        "Expected:\n"
        "${MNN_INCLUDE_DIR}/MNN/Interpreter.hpp"
    )

endif()

message(STATUS "MNN root: ${MNN_ROOT}")
message(STATUS "MNN include: ${MNN_INCLUDE_DIR}")

set(MNN_LIB
    "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libMNN.so"
)

if(NOT EXISTS "${MNN_LIB}")

    message(FATAL_ERROR
        "libMNN.so NOT FOUND:\n"
        "${MNN_LIB}"
    )

endif()

message(STATUS "MNN library: ${MNN_LIB}")

add_library(
    MNN
    SHARED
    IMPORTED
)

set_target_properties(
    MNN
    PROPERTIES
    IMPORTED_LOCATION
        "${MNN_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES
        "${MNN_INCLUDE_DIR}"
)

add_library(
    sana_native
    SHARED
    sana_jni.cpp
)

target_include_directories(
    sana_native
    PRIVATE
    "${MNN_INCLUDE_DIR}"
)

find_library(
    log-lib
    log
)

find_library(
    android-lib
    android
)

target_link_libraries(
    sana_native

    MNN

    ${log-lib}

    ${android-lib}
)

message(STATUS "======================================")
message(STATUS "MNN CONFIGURATION COMPLETE")
message(STATUS "======================================")
