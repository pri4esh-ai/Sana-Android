cmake_minimum_required(VERSION 3.22.1)

project(sana_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# PROJECT PATHS
# ============================================================

get_filename_component(
    APP_CPP_DIR
    "${CMAKE_CURRENT_LIST_DIR}"
    ABSOLUTE
)

get_filename_component(
    APP_DIR
    "${APP_CPP_DIR}/../.."
    ABSOLUTE
)

get_filename_component(
    PROJECT_ROOT
    "${APP_DIR}/../.."
    ABSOLUTE
)

set(
    MNN_ROOT
    "${PROJECT_ROOT}/third_party/MNN"
)

set(
    MNN_INCLUDE_DIR
    "${MNN_ROOT}/include"
)

set(
    MNN_HEADER
    "${MNN_INCLUDE_DIR}/MNN/Interpreter.hpp"
)

set(
    MNN_LIB
    "${PROJECT_ROOT}/app/src/main/jniLibs/${ANDROID_ABI}/libMNN.so"
)

message(STATUS "======================================")
message(STATUS "SANA / MNN CONFIGURATION")
message(STATUS "======================================")

message(STATUS "APP_CPP_DIR:")
message(STATUS "${APP_CPP_DIR}")

message(STATUS "PROJECT_ROOT:")
message(STATUS "${PROJECT_ROOT}")

message(STATUS "MNN_ROOT:")
message(STATUS "${MNN_ROOT}")

message(STATUS "MNN_INCLUDE_DIR:")
message(STATUS "${MNN_INCLUDE_DIR}")

message(STATUS "MNN_HEADER:")
message(STATUS "${MNN_HEADER}")

message(STATUS "MNN_LIB:")
message(STATUS "${MNN_LIB}")

# ============================================================
# VERIFY HEADER
# ============================================================

if(NOT EXISTS "${MNN_HEADER}")

    message(FATAL_ERROR
        "MNN HEADER NOT FOUND:\n${MNN_HEADER}"
    )

endif()

message(STATUS "MNN HEADER FOUND")

# ============================================================
# VERIFY LIBRARY
# ============================================================

if(NOT EXISTS "${MNN_LIB}")

    message(FATAL_ERROR
        "MNN LIBRARY NOT FOUND:\n${MNN_LIB}"
    )

endif()

message(STATUS "MNN LIBRARY FOUND")

# ============================================================
# IMPORT MNN
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
)

# ============================================================
# SANA JNI
# ============================================================

add_library(
    sana_native
    SHARED
    sana_jni.cpp
)

# ============================================================
# CRITICAL INCLUDE PATH
# ============================================================

target_include_directories(
    sana_native
    PRIVATE
        "${MNN_INCLUDE_DIR}"
)

# ============================================================
# ANDROID LIBRARIES
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
# LINK
# ============================================================

target_link_libraries(
    sana_native
    MNN
    ${log-lib}
    ${android-lib}
)

message(STATUS "======================================")
message(STATUS "MNN CONFIGURATION COMPLETE")
message(STATUS "======================================")
