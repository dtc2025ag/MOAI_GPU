cmake_minimum_required(VERSION 3.20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # compile files

# Must use CUDA 11
if (UNIX)
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc")
endif()

project(MOAI-GPU LANGUAGES CXX CUDA)

# CMAKE config
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

# Debugging config
option(SHOW_WARNINGS "Enable warnings" OFF)
if (SHOW_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-comment")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra,-Wpedantic,-Wno-comment")
endif()

# Language config
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA config
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES native) # auto-detect GPU
endif()
set(CMAKE_CUDA_ARCHITECTURES 80 90)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 --default-stream per-thread")

# Print CUDA info
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Compiler ID: ${CMAKE_CUDA_COMPILER_ID}")
message(STATUS "CUDA Compiler Version: ${CMAKE_CUDA_COMPILER_VERSION}")

# PhantomFHE config
option(PHANTOM_USE_CUDA_PTX "Use CUDA PTX Assembly" ON)
message(STATUS "Use CUDA PTX Assembly: ${PHANTOM_USE_CUDA_PTX}")
if (PHANTOM_USE_CUDA_PTX)
    add_compile_definitions(PHANTOM_USE_CUDA_PTX)
endif()

# PhantomFHE
include_directories(thirdparty/phantom-fhe/include)


file(GLOB_RECURSE MOAI_CUDA_SOURCE_FILES CONFIGURE_DEPENDS
  "src/test.cu"
  "src/include/source/matrix_mul/Ct_pt_matrix_mul.cuh"
  "src/include/source/matrix_mul/Batch_encode_encrypt.cuh"
  "src/include/source/matrix_mul/Ct_ct_matrix_mul.cuh"
  "src/include/source/non_linear_func/gelu_other.cuh"
  "src/include/source/non_linear_func/layernorm.cuh"
  "src/include/source/non_linear_func/softmax.cuh"
  "src/include/test/test_phantom_ckks.cuh"
  "src/include/test/matrix_mul/test_batch_encode_encrypt.hpp"
  "src/include/test/matrix_mul/test_ct_pt_matrix_mul.cu"
  "src/include/test/matrix_mul/test_ct_ct_matrix_mul.cu"
  "src/include/test/non_linear_func/test_gelu.cuh"
  "src/include/test/non_linear_func/test_layernorm.cuh"
  "src/include/test/bootstrapping/bootstrapping.cuh"
  "src/include/test/non_linear_func/test_softmax.cuh"
  "src/include/test/test_single_layer.cuh"
  "src/include/source/utils.cuh"
  "src/include/source/utils_moai.cuh"
  # "src/bootstrapping/*.cu"
  # "src/bootstrapping/*.cu"
  # "src/source/bootstrapping.cu"
  "src/include/source/bootstrapping/*.cu"
  "src/include/source/ckks_evaluator_parallel.cu"
  "thirdparty/phantom-fhe/src/*.cu"
)



add_executable(test ${MOAI_CUDA_SOURCE_FILES})

find_package(OpenMP REQUIRED)

target_link_libraries(test PRIVATE OpenMP::OpenMP_CXX)

target_compile_options(test PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
)
# target_compile_options(test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>)


target_link_libraries(test PRIVATE ntl gmp m pthread)
target_include_directories(test PRIVATE
  /usr/local/include
  /usr/local/include/NTL
  # src/bootstrapping/common
  # src/bootstrapping
  # scratch
  # scratch/bootstrapping/common
  # scratch/bootstrapping
  ${CMAKE_SOURCE_DIR}/src/include/source/bootstrapping/common
  ${CMAKE_SOURCE_DIR}/src/include/source/bootstrapping
  ${CMAKE_SOURCE_DIR}/src/include
)



file(COPY "${CMAKE_SOURCE_DIR}/src/data/" DESTINATION "${CMAKE_BINARY_DIR}/")


