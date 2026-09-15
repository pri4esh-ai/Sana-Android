cmake_minimum_required(VERSION 3.22.1)

project(sana_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(sana_native SHARED sana_jni.cpp)

find_library(log-lib log)
find_library(android-lib android)

target_link_libraries(
    sana_native
    ${log-lib}
    ${android-lib}
)
