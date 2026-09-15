cmake_minimum_required(VERSION 3.22.1)

project(sana_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MNN_DIR
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/MNN
)

set(MNN_INCLUDE_DIR
        ${MNN_DIR}/source
)

add_library(
        sana_native
        SHARED
        sana_jni.cpp
)

target_include_directories(
        sana_native
        PRIVATE
        ${MNN_INCLUDE_DIR}
)

add_library(
        MNN
        SHARED
        IMPORTED
)

set_target_properties(
        MNN
        PROPERTIES
        IMPORTED_LOCATION
        ${MNN_DIR}/build_android/libMNN.so
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
