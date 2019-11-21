cmake_minimum_required(VERSION 2.8.3)
#set( CMAKE_CXX_FLAGS "-std=c++14" )

project(example)

SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++14 -fopenmp -pthread")

set(BOOST_ROOT "/usr/include/boost") # either set it here or from the command line  
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost REQUIRED COMPONENTS system)
find_package( OpenCV REQUIRED )
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  message(STATUS "OPENMP FOUND")
  set(OpenMP_FLAGS ${OpenMP_CXX_FLAGS})  # or if you use C: ${OpenMP_C_FLAGS}
  set(OpenMP_LIBS gomp)
endif()

include_directories( "/usr/include/eigen3" )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(bilateral bilateral.cpp)
target_include_directories(bilateral PUBLIC ${Boost_INCLUDE_DIRS}) 
target_link_libraries(bilateral ${Boost_LIBRARIES})
target_link_libraries(bilateral stdc++fs)
target_link_libraries(bilateral ${OpenCV_LIBS})
target_link_libraries(bilateral ${OpenMP_LIBS})

