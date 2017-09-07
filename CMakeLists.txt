cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

get_filename_component(ProjectName ${CMAKE_SOURCE_DIR} NAME)
string(REPLACE " " "_" ProjectName ${ProjectName})

message(STATUS "configuring project: " ${ProjectName})

project(${ProjectName})

find_package(PCL 1.8 COMPONENTS io REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#cnpy
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/cnpy)
set(CNPY_SRC ${CMAKE_SOURCE_DIR}/3rdparty/cnpy/cnpy.cpp)

#zlib
set(ZLIB_ROOT ${CMAKE_SOURCE_DIR}/3rdparty/zlib/install CACHE PATH "root of zlib folder containing ZLIB_ROOT/include and ZLIB_ROOT/lib")
include_directories(${ZLIB_ROOT}/include)
if(WIN32)
    set(ZLIB_STATICLIB ${ZLIB_ROOT}/lib/zlibstatic.lib)
else()
    set(ZLIB_STATICLIB ${ZLIB_ROOT}/lib/zlibstatic.a)
endif()

add_executable(${ProjectName} ${ProjectName}.cpp ${CNPY_SRC})
target_link_libraries(${ProjectName} ${PCL_LIBRARIES} ${ZLIB_STATICLIB})
#set_target_properties(${ProjectName} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE "C:/lib/build/vs2015/pcl/bin")
