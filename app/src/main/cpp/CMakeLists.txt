cmake_minimum_required(VERSION 3.22.1)

project("sana_native")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MNN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/mnn)

add_library(
        sana_native
        SHARED
        sana_jni.cpp
)

find_library(
        log-lib
        log
)

find_library(
        android-lib
        android
)

target_include_directories(
        sana_native
        PRIVATE
        ${MNN_ROOT}/include
)

target_link_libraries(
        sana_native
        ${log-lib}
        ${android-lib}
)a
