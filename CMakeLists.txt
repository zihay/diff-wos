cmake_minimum_required(VERSION 3.15)

project(wos-cuda VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
check_language(CUDA)
message(STATUS "CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CXX_STANDARD 17)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    cmake_policy(SET CMP0104 OLD) # for CUDA_ARCHITECTURES
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr -std=c++17")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

# pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})

# Locate pybind11
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c
          "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE _tmp_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
find_package(pybind11 CONFIG REQUIRED)

#eigen
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ext/eigen)
# lbvh
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ext/lbvh)

# drjit
set(DRJIT_USE_LOCAL OFF)
if(DRJIT_USE_LOCAL)
  option(DRJIT_ENABLE_JIT "" ON)
  option(DRJIT_ENABLE_AUTODIFF "" ON)
  option(DRJIT_ENABLE_PYTHON "" ON)
  add_subdirectory(ext/drjit)
else()
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import drjit;print(drjit.get_cmake_dir())" OUTPUT_VARIABLE drjit_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(${drjit_DIR})
  find_package(drjit)
  if (NOT ${drjit_FOUND})
    message(FATAL_ERROR "Dr.Jit not found. Please install Dr.Jit using \"pip install drjit\"")
  endif()
endif()

include_directories(include)

add_subdirectory(src)