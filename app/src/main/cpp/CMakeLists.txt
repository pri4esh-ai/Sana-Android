cmake_minimum_required(VERSION 3.22.1)

project(sana_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "======================================")
message(STATUS "SANA NATIVE BUILD")
message(STATUS "======================================")
message(STATUS "ANDROID ABI: ${ANDROID_ABI}")

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

get_filename_component(PROJECT_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../"
    ABSOLUTE
)

set(MNN_ROOT
    "${PROJECT_ROOT}/third_party/MNN"
)

# ---------------------------------------------------------
# Locate MNN headers
# ---------------------------------------------------------

set(MNN_HEADER_CANDIDATES
    "${MNN_ROOT}/include"
    "${MNN_ROOT}/source"
    "${MNN_ROOT}/3rd_party"
    "${MNN_ROOT}"
)

set(MNN_INCLUDE_DIR "")

foreach(CANDIDATE ${MNN_HEADER_CANDIDATES})

    if(EXISTS "${CANDIDATE}/MNN/Interpreter.hpp")
        set(MNN_INCLUDE_DIR "${CANDIDATE}")
        break()
    endif()

endforeach()

if(MNN_INCLUDE_DIR STREQUAL "")

    message(FATAL_ERROR
        "MNN/Interpreter.hpp was not found.\n"
        "MNN root: ${MNN_ROOT}"
    )

endif()

message(STATUS "MNN root: ${MNN_ROOT}")
message(STATUS "MNN include: ${MNN_INCLUDE_DIR}")

# ---------------------------------------------------------
# Prebuilt MNN library
# ---------------------------------------------------------

set(MNN_LIB
    "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libMNN.so"
)

if(NOT EXISTS "${MNN_LIB}")

    message(FATAL_ERROR
        "MNN shared library not found:\n"
        "${MNN_LIB}\n\n"
        "The GitHub Actions workflow must copy libMNN.so "
        "into app/src/main/jniLibs/arm64-v8a/"
    )

endif()

message(STATUS "MNN library: ${MNN_LIB}")

# ---------------------------------------------------------
# Imported MNN shared library
# ---------------------------------------------------------

add_library(MNN SHARED IMPORTED)

set_target_properties(
    MNN
    PROPERTIES
    IMPORTED_LOCATION "${MNN_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${MNN_INCLUDE_DIR}"
)

# ---------------------------------------------------------
# Sana JNI
# ---------------------------------------------------------

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

# ---------------------------------------------------------
# Android libraries
# ---------------------------------------------------------

find_library(log-lib log)
find_library(android-lib android)

# ---------------------------------------------------------
# Link
# ---------------------------------------------------------

target_link_libraries(
    sana_native
    MNN
    ${log-lib}
    ${android-lib}
)

message(STATUS "======================================")
message(STATUS "MNN CMAKE CONFIGURATION COMPLETE")
message(STATUS "======================================")
