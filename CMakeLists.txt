# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Jiaxin Peng (jiaxinpeng.nudt AT gmail.com)

cmake_minimum_required(VERSION 3.2)

project(YH3DR LANGUAGES CXX)

#set version
set(YH3DR_VERSION "1.0.0")


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

################################################################################
# Include CMake dependencies
################################################################################

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(CheckCXXCompilerFlag)

# Include helper macros and commands, and allow the included file to override
# the CMake policies in this file
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeHelper.cmake NO_POLICY_SCOPE)


################################################################################
# Options
################################################################################

option(SIMD_ENABLED "Whether to enable SIMD optimizations" ON)
option(OPENMP_ENABLED "Whether to enable OpenMP parallelization" ON)
option(IPO_ENABLED "Whether to enable interprocedural optimization" ON)
option(CUDA_ENABLED "Whether to enable CUDA, if available" ON)
option(GUI_ENABLED "Whether to enable the graphical UI" ON)
option(OPENGL_ENABLED "Whether to enable OpenGL, if available" ON)
option(TESTS_ENABLED "Whether to build test binaries" OFF)
option(ASAN_ENABLED "Whether to enable AddressSanitizer flags" OFF)
option(PROFILING_ENABLED "Whether to enable google-perftools linker flags" OFF)
option(CCACHE_ENABLED "Whether to enable compiler caching, if available" ON)
option(CGAL_ENABLED "Whether to enable the CGAL library" ON)
option(BOOST_STATIC "Whether to enable static boost library linker flags" ON)

if(TESTS_ENABLED)
    enable_testing()
endif()

if(BOOST_STATIC)
    set(Boost_USE_STATIC_LIBS ON)
else()
    add_definitions("-DBOOST_TEST_DYN_LINK")
endif()

# Build position-independent code, so that shared libraries can link against
# COLMAP's static libraries.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

################################################################################
# Find packages
################################################################################

find_package(MPI REQUIRED)

if(OPENMP_ENABLED)
    find_package(OpenMP QUIET)
endif()


find_package(Boost REQUIRED COMPONENTS
             program_options
             filesystem
             graph
             mpi
             system
             unit_test_framework)

find_package(Eigen3 REQUIRED)


find_package(FLANN REQUIRED)
find_package(LZ4 REQUIRED)

find_package(Metis REQUIRED)

find_package(Glog REQUIRED)

find_package(OpenCV REQUIRED)

set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)
find_package(Glew REQUIRED)
find_package(Git)

################################################################################
# Compiler specific configuration
################################################################################

if(CMAKE_BUILD_TYPE)
    message(STATUS "Build type specified as ${CMAKE_BUILD_TYPE}")
else()
    message(STATUS "Build type not specified, using Release")
    set(CMAKE_BUILD_TYPE Release)
    set(IS_DEBUG OFF)
endif()

if(IS_MSVC)
    # Some fixes for the Glog library.
    add_definitions("-DGLOG_NO_ABBREVIATED_SEVERITIES")
    add_definitions("-DGL_GLEXT_PROTOTYPES")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
    # Enable object level parallel builds in Visual Studio.
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

if(IS_GNU)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
        message(FATAL_ERROR "GCC version 4.8 or older not supported")
    endif()

    # Hide incorrect warnings for uninitialized Eigen variables under GCC.
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-maybe-uninitialized")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized")
endif()

if(IS_DEBUG)
    add_definitions("-DEIGEN_INITIALIZE_MATRICES_BY_NAN")
endif()

if(SIMD_ENABLED)
    message(STATUS "Enabling SIMD support")
else()
    message(STATUS "Disabling SIMD support")
endif()

if(OPENMP_ENABLED AND OPENMP_FOUND)
    message(STATUS "Enabling OpenMP support")
    add_definitions("-DOPENMP_ENABLED")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(STATUS "Disabling OpenMP support")
endif()

if(IPO_ENABLED AND NOT IS_DEBUG AND NOT IS_GNU)
    message(STATUS "Enabling interprocedural optimization")
    set_property(DIRECTORY PROPERTY INTERPROCEDURAL_OPTIMIZATION 1)
else()
    message(STATUS "Disabling interprocedural optimization")
endif()

if(ASAN_ENABLED)
    message(STATUS "Enabling ASan support")
    if(IS_CLANG OR IS_GNU)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer -fsanitize-address-use-after-scope)
        add_link_options(-fsanitize=address)
    else()
        message(FATAL_ERROR "Unsupported compiler for ASan mode")
    endif()
endif()

if(CUDA_ENABLED AND CUDA_FOUND)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        message(
            FATAL_ERROR "You must set CMAKE_CUDA_ARCHITECTURES to e.g. 'native', 'all-major', '70', etc. "
            "More information at https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html")
    endif()

    add_definitions("-DCUDA_ENABLED")

    # Fix for some combinations of CUDA and GCC (e.g. under Ubuntu 16.04).
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_FORCE_INLINES")
    # Do not show warnings if the architectures are deprecated.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
    # Do not show warnings if cuda library functions are deprecated.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-declarations")
    # Explicitly set PIC flags for CUDA targets.
    if(NOT IS_MSVC)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -fPIC")
    endif()

    message(STATUS "Enabling CUDA support (version: ${CUDAToolkit_VERSION}, "
                    "archs: ${CMAKE_CUDA_ARCHITECTURES})")
else()
    set(CUDA_ENABLED OFF)
    message(STATUS "Disabling CUDA support")
endif()

if(OPENGL_ENABLED)
    add_definitions("-DOPENGL_ENABLED")
    message(STATUS "Enabling OpenGL support")
else()
    message(STATUS "Disabling OpenGL support")
endif()

if(CCACHE_ENABLED)
    find_program(CCACHE ccache)
    if(CCACHE)
        message(STATUS "Enabling ccache support")
        set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE})
        set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
    else()
        message(STATUS "Disabling ccache support")
    endif()
else()
    message(STATUS "Disabling ccache support")
endif()

if(PROFILING_ENABLED)
    message(STATUS "Enabling profiling support")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lprofiler -ltcmalloc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lprofiler -ltcmalloc")
else()
    message(STATUS "Disabling profiling support")
endif()

if(CGAL_FOUND AND CGAL_ENABLED)
    message(STATUS "Enabling CGAL support")
    add_definitions("-DCGAL_ENABLED")
else()
    message(STATUS "Disabling CGAL support")
    set(CGAL_ENABLED OFF)
endif()



if (MPI_FOUND)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
    list(APPEND COLMAP_EXTERNAL_LIBRARIES ${MPI_LIBRARIES})
else (MPI_FOUND)
    message(SEND_ERROR "This application cannot compile without MPI")
endif (MPI_FOUND)

################################################################################
# Add sources
################################################################################
# Generate source file with version definitions.
#include(GenerateVersionDefinitions)


set(YH3DR_INCLUDE_DIRS "./")

set(COLMAP_INCLUDE_DIRS
    ${Boost_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${FLANN_INCLUDE_DIRS}
    ${LZ4_INCLUDE_DIRS}
    ${FREEIMAGE_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${METIS_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${SQLite3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${YH3DR_INCLUDE_DIRS}
)

message(STATUS, "BOOST include dirs: ${Boost_INCLUDE_DIRS}")

set(COLMAP_LINK_DIRS
    ${Boost_LIBRARY_DIRS}
)


set(COLMAP_EXTERNAL_LIBRARIES
    ${CMAKE_DL_LIBS}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_MPI_LIBRARY}#ADD MPI Libraries for LINKING
    ${GLOG_LIBRARIES}
    ${FLANN_LIBRARIES}
    ${LZ4_LIBRARIES}
    ${FREEIMAGE_LIBRARIES}
    ${METIS_LIBRARIES}
    ${CERES_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${OpenCV_LIBS}
    ${SQLite3_LIBRARIES}
)

set(YH3DR_SOURCES
    xsfm.cpp
    #sfm/feature/UMatching.cpp
)

if(OPENMP_FOUND)
    list(APPEND COLMAP_EXTERNAL_LIBRARIES ${OpenMP_libomp_LIBRARY})
endif()


if(CGAL_FOUND)
    list(APPEND COLMAP_INCLUDE_DIRS ${CGAL_INCLUDE_DIRS} ${GMP_INCLUDE_DIR})
    list(APPEND COLMAP_EXTERNAL_LIBRARIES ${CGAL_LIBRARY} ${GMP_LIBRARIES})
    list(APPEND COLMAP_LINK_DIRS ${CGAL_LIBRARIES_DIR})
endif()

if(UNIX)
    list(APPEND COLMAP_EXTERNAL_LIBRARIES pthread)
endif()


set(COLMAP_INTERNAL_LIBRARIES
    lsd
    pba
    poisson_recon
    vlfeat
    TinyEXIF
    mvg
)

include_directories(
    libmvg
    sfm
    ${COLMAP_INCLUDE_DIRS}
    
)

message(STATUS ${COLMAP_LINK_DIRS})
message(STATUS ${YH3DR_SOURCES})
message(STATUS "----------------------------------")

link_directories(${COLMAP_LINK_DIRS})

add_subdirectory(libmvg)
add_subdirectory(sfm)
add_subdirectory(mvs)
add_subdirectory(tools)
add_subdirectory(model)
add_subdirectory(thirdParty)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive -lboost_mpi -lopencv_core")


add_executable(xsfm_exe
    ${YH3DR_SOURCES}
    sfm/feature/UImage.cpp
    sfm/feature/UCamera.cpp
    #sfm/feature/UImageMatching.cpp
    sfm/feature/UMatching.cpp
    
    )
set_target_properties(xsfm_exe PROPERTIES OUTPUT_NAME xsfm)

target_link_libraries(xsfm_exe
        ${COLMAP_INTERNAL_LIBRARIES} ${COLMAP_EXTERNAL_LIBRARIES})
        message(STATUS "${COLMAP_INTERNAL_LIBRARIES}")

COLMAP_ADD_SOURCES(
    xsfm.h xsfm.cpp
)

