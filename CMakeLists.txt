cmake_minimum_required(VERSION 3.21)

#set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")

set(PROJECT decolor_metrics)
project(${PROJECT})

#link_directories(/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib)
#
#include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
#link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(nlohmann_json REQUIRED)

add_subdirectory(utils)

set(CMAKE_CXX_STANDARD 17)

#set(MAIN main.cpp)
#set(MAIN main2.cpp)
#set(MAIN main_method.cpp)
#set(MAIN main_ame.cpp)
#set(MAIN main_new.cpp)
set(MAIN metric-json.cpp)

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include )
link_directories( ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)


add_executable(${PROJECT} ${MAIN} DecolorMeasurer.cpp decolor.cpp AME.cpp)

target_link_libraries(${PROJECT} ${OpenCV_LIBS} utils nlohmann_json::nlohmann_json)