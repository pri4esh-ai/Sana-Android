cmake_minimum_required(VERSION 3.22.1)

project(sana_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "======================================")
message(STATUS "SANA NATIVE BUILD")
message(STATUS "======================================")
message(STATUS "ANDROID ABI: ${ANDROID_ABI}")

# ============================================================
# Find Sana-Android project root
#
# Current file:
# app/src/main/cpp/CMakeLists.txt
#
# Project root:
# Sana-Android/
# ============================================================

get_filename_component(
    PROJECT_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../"
    ABSOLUTE
)

set(MNN_ROOT
    "${PROJECT_ROOT}/third_party/MNN"
)

message(STATUS "PROJECT ROOT: ${PROJECT_ROOT}")
message(STATUS "MNN ROOT: ${MNN_ROOT}")

# ============================================================
# MNN HEADER DIRECTORY
#
# IMPORTANT:
#
# MNN/Interpreter.hpp is expected at:
#
# third_party/MNN/include/MNN/Interpreter.hpp
#
# NOT:
#
# third_party/MNN/source/MNN/Interpreter.hpp
# ============================================================

set(MNN_INCLUDE_DIR
    "${MNN_ROOT}/include"
)

message(STATUS "MNN INCLUDE DIR: ${MNN_INCLUDE_DIR}")

# ============================================================
# Verify MNN headers
# ============================================================

if(NOT EXISTS
    "${MNN_INCLUDE_DIR}/MNN/Interpreter.hpp"
)

    message(FATAL_ERROR
        "\n"
        "======================================\n"
        "MNN HEADER ERROR\n"
        "======================================\n"
        "Expected header:\n"
        "${MNN_INCLUDE_DIR}/MNN/Interpreter.hpp\n"
        "\n"
        "MNN root:\n"
        "${MNN_ROOT}\n"
        "\n"
        "Please verify the MNN checkout.\n"
        "======================================\n"
    )

endif()

message(STATUS
    "MNN Interpreter.hpp FOUND:"
)

message(STATUS
    "${MNN_INCLUDE_DIR}/MNN/Interpreter.hpp"
)

# ============================================================
# MNN shared library
#
# Workflow installs:
#
# app/src/main/jniLibs/arm64-v8a/libMNN.so
# ============================================================

set(MNN_LIB
    "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libMNN.so"
)

message(STATUS "MNN LIBRARY: ${MNN_LIB}")

if(NOT EXISTS "${MNN_LIB}")

    message(FATAL_ERROR
        "\n"
        "======================================\n"
        "MNN LIBRARY ERROR\n"
        "======================================\n"
        "Expected:\n"
        "${MNN_LIB}\n"
        "\n"
        "libMNN.so was not found.\n"
        "======================================\n"
    )

endif()

message(STATUS "libMNN.so FOUND")

# ============================================================
# Imported MNN shared library
# ============================================================

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

# ============================================================
# Sana native JNI library
# ============================================================

add_library(
    sana_native
    SHARED

    sana_jni.cpp
)

# ============================================================
# MNN include path
#
# This is the critical fix.
# ============================================================

target_include_directories(
    sana_native
    PRIVATE

    "${MNN_INCLUDE_DIR}"
)

# ============================================================
# Android libraries
# ============================================================

find_library(
    log-lib
    log
)

find_library(
    android-lib
    android
)

# ============================================================
# Link Sana native library
# ============================================================

target_link_libraries(
    sana_native

    MNN

    ${log-lib}

    ${android-lib}
)

# ============================================================
# Final diagnostics
# ============================================================

message(STATUS "======================================")
message(STATUS "SANA NATIVE CONFIGURATION COMPLETE")
message(STATUS "======================================")

message(STATUS
    "MNN include:"
)

message(STATUS
    "${MNN_INCLUDE_DIR}"
)

message(STATUS
    "MNN library:"
)

message(STATUS
    "${MNN_LIB}"
)

message(STATUS
    "======================================")
