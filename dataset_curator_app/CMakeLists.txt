cmake_minimum_required(VERSION 3.16)
project(curator_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(GDAL REQUIRED)

add_executable(curator_app
    main.cpp
    mainwindow.h
    mainwindow.cpp
    gridviewerwidget.h
    gridviewerwidget.cpp
)

target_include_directories(curator_app PRIVATE ${GDAL_INCLUDE_DIR})
target_link_libraries(curator_app Qt6::Widgets ${GDAL_LIBRARY})

