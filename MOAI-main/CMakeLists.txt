project(hexl_example LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 14)

include_directories(include)
include_directories(/usr/include)
include_directories(/usr/local/include)

link_directories(/usr/lib)
link_directories(/usr/local/lib) 

set(CMAKE_C_COMPILER "gcc-14")
set(CMAKE_CPP_COMPILER "g++-14")

include(FindPkgConfig)
if(NOT PKG_CONFIG_FOUND)
  message(FATAL_ERROR "pkg-config not found!" )
endif()

set(CMAKE_CXX_FLAGS "-lstdc++ -O3 -fopenmp")
set(CMAKE_C_FLAGS "-lstdc++ -O3 -fopenmp")


find_package(SEAL 4.1 REQUIRED)

add_subdirectory(include/source/bootstrapping/common)
add_subdirectory(include/source/bootstrapping)

add_executable(test test.cpp
  ${CMAKE_SOURCE_DIR}/include/source/bootstrapping/Bootstrapper.cpp
  ${CMAKE_SOURCE_DIR}/include/source/bootstrapping/ModularReducer.cpp
  ${CMAKE_SOURCE_DIR}/include/source/bootstrapping/common/Choosemax.cpp
  ${CMAKE_SOURCE_DIR}/include/source/bootstrapping/common/MinicompFunc.cpp
  ${CMAKE_SOURCE_DIR}/include/source/bootstrapping/common/MinicompRemez.cpp
  ${CMAKE_SOURCE_DIR}/include/source/bootstrapping/common/Point.cpp
  ${CMAKE_SOURCE_DIR}/include/source/bootstrapping/common/PolyUpdate.cpp
  ${CMAKE_SOURCE_DIR}/include/source/bootstrapping/common/func.cpp
  ${CMAKE_SOURCE_DIR}/include/source/bootstrapping/common/Polynomial.cpp
  ${CMAKE_SOURCE_DIR}/include/source/bootstrapping/common/Remez.cpp
  ${CMAKE_SOURCE_DIR}/include/source/bootstrapping/common/RemezApp.cpp
)

target_include_directories(test PUBLIC
    /usr/local/include
    /usr/local/include/NTL
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/source
    ${CMAKE_SOURCE_DIR}/include/source/bootstrapping
    ${CMAKE_SOURCE_DIR}/include/source/bootstrapping/common
)

target_link_libraries(test crypto ntl gmp m pthread SEAL::seal)

file(COPY "${CMAKE_SOURCE_DIR}/data/" DESTINATION "${CMAKE_BINARY_DIR}/")
