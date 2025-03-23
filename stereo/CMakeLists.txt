cmake_minimum_required(VERSION 3.0.2)

#项目名称
project(Test)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( Test  dis.cpp)
target_link_libraries( Test ${OpenCV_LIBS} )






