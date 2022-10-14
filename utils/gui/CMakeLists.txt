cmake_minimum_required(VERSION 3.14)

project(gui)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

#set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake)

set(CMAKE_PREFIX_PATH /usr/local/Cellar/qt@5/5.15.2/lib/cmake/Qt5)

find_package(Qt5 COMPONENTS Widgets Concurrent Test REQUIRED)
find_package(OpenCV REQUIRED)

add_library(gui window_QT.cpp)

target_link_libraries(gui ${OpenCV_LIBS} Qt5::Widgets Qt5::Concurrent Qt5::Test)
