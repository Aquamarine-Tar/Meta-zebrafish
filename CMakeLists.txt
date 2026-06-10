cmake_minimum_required(VERSION 3.5)
project(metaworm LANGUAGES C CXX CUDA)

# CUDA 环境设置
find_package(CUDAToolkit QUIET)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda -allow-unsupported-compiler")

add_compile_options(-fPIC)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-std=gnu++11>)
add_compile_options($<$<COMPILE_LANGUAGE:C>:-std=gnu11>)
add_compile_options(-Wdeprecated-declarations)

add_definitions(-DSOFTCON_DIR="${PROJECT_SOURCE_DIR}")
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_module")
option(METAWORM_ENABLE_EWORM "Build C302/eworm Python interaction support" OFF)
if(METAWORM_ENABLE_EWORM)
  add_definitions(-DMETAWORM_ENABLE_EWORM)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

add_subdirectory( sim )
add_subdirectory( render )
if(METAWORM_ENABLE_EWORM)
  add_subdirectory( interact )
endif()

set(META_WORM_INCLUDES 
    ${WORM_RENDER_INCLUDES}
    ${WORM_INTERACT_INCLUDES}
    ${WORM_SIM_INCLUDES} PARENT_SCOPE)

file(COPY args data DESTINATION ${CMAKE_BINARY_DIR})
