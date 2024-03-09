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
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

# Find package module for COLMAP library.
#
# The following variables are set by this module:
#
#   COLMAP_FOUND: TRUE if COLMAP is found.
#   COLMAP_VERSION: COLMAP version.
#   COLMAP_INCLUDE_DIRS: Include directories for COLMAP.
#   COLMAP_LINK_DIRS: Link directories for COLMAP.
#   COLMAP_LIBRARIES: Libraries required to link COLMAP.
#   COLMAP_CUDA_ENABLED: Whether COLMAP was compiled with CUDA support.
#   COLMAP_GUI_ENABLED: Whether COLMAP was compiled with the graphical UI.
#   COLMAP_CGAL_ENABLED: Whether COLMAP was compiled with CGAL dependencies.

get_filename_component(COLMAP_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_FILE} PATH)
set(COLMAP_INSTALL_PREFIX "${COLMAP_INSTALL_PREFIX}/../..")

set(COLMAP_FOUND FALSE)

# Set hints for finding dependency packages.

set(EIGEN3_INCLUDE_DIR_HINTS )

set(FLANN_INCLUDE_DIR_HINTS )
set(FLANN_LIBRARY_DIR_HINTS )

set(LZ4_INCLUDE_DIR_HINTS )
set(LZ4_LIBRARY_DIR_HINTS )

set(FREEIMAGE_INCLUDE_DIR_HINTS )
set(FREEIMAGE_LIBRARY_DIR_HINTS )

set(METIS_INCLUDE_DIR_HINTS )
set(METIS_LIBRARY_DIR_HINTS )

set(GLEW_INCLUDE_DIR_HINTS )
set(GLEW_LIBRARY_DIR_HINTS )

set(GLOG_INCLUDE_DIR_HINTS )
set(GLOG_LIBRARY_DIR_HINTS )

set(SQLite3_INCLUDE_DIR_HINTS )
set(SQLite3_LIBRARY_DIR_HINTS )

# Find dependency packages.

set(TEMP_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH ${COLMAP_INSTALL_PREFIX}/share/colmap/cmake)

if(COLMAP_FIND_QUIETLY)
    find_package(Ceres QUIET)

    find_package(Boost COMPONENTS
                program_options
                filesystem
                system
                unit_test_framework
                QUIET)

    find_package(Eigen3 QUIET)

    find_package(FLANN QUIET)
    find_package(LZ4 QUIET)

    find_package(FreeImage QUIET)

    find_package(Metis QUIET)

    find_package(Glog QUIET)

    find_package(SQLite3 QUIET)

    find_package(OpenGL QUIET)
    find_package(Glew QUIET)
else()
    find_package(Ceres REQUIRED)

    find_package(Boost COMPONENTS
                program_options
                filesystem
                system
                unit_test_framework
                REQUIRED)

    find_package(Eigen3 REQUIRED)

    find_package(FLANN REQUIRED)
    find_package(LZ4 REQUIRED)

    find_package(FreeImage REQUIRED)

    find_package(Metis REQUIRED)

    find_package(Glog REQUIRED)

    find_package(SQLite3 REQUIRED)

    find_package(OpenGL REQUIRED)
    find_package(Glew REQUIRED)
endif()

# Set the exported variables.

set(COLMAP_FOUND TRUE)

set(COLMAP_VERSION 3.9)

set(COLMAP_OPENMP_ENABLED ON)

set(COLMAP_CUDA_ENABLED OFF)
set(COLMAP_CUDA_MIN_VERSION 7.0)

set(COLMAP_GUI_ENABLED OFF)

set(COLMAP_CGAL_ENABLED OFF)

set(COLMAP_INCLUDE_DIRS
    ${COLMAP_INSTALL_PREFIX}/include/
    ${COLMAP_INSTALL_PREFIX}/include/colmap
    ${COLMAP_INSTALL_PREFIX}/include/colmap/lib
    ${Boost_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${FLANN_INCLUDE_DIRS}
    ${LZ4_INCLUDE_DIRS}
    ${FREEIMAGE_INCLUDE_DIRS}
    ${METIS_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${SQLite3_INCLUDE_DIRS}
)

set(COLMAP_LINK_DIRS
    ${COLMAP_INSTALL_PREFIX}/lib/colmap
    ${Boost_LIBRARY_DIRS}
)

set(COLMAP_INTERNAL_LIBRARIES
    lsd
    pba
    poisson_recon
    sqlite3
    sift_gpu
    vlfeat
)

set(COLMAP_EXTERNAL_LIBRARIES
    ${CMAKE_DL_LIBS}
    ${GLOG_LIBRARIES}
    ${FLANN_LIBRARIES}
    ${LZ4_LIBRARIES}
    ${FREEIMAGE_LIBRARIES}
    ${METIS_LIBRARIES}
    ${CERES_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${SQLite3_LIBRARIES}
)

if(UNIX)
    list(APPEND COLMAP_EXTERNAL_LIBRARIES
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_PROGRAM_OPTIONS_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
        pthread)
endif()

if(COLMAP_OPENMP_ENABLED)
    find_package(OpenMP QUIET)
    add_definitions("-DOPENMP_ENABLED")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    list(APPEND COLMAP_EXTERNAL_LIBRARIES ${OpenMP_libomp_LIBRARY})
endif()

if(COLMAP_CUDA_ENABLED)
    find_package(CUDA ${COLMAP_CUDA_MIN_VERSION} QUIET)
    list(APPEND COLMAP_EXTERNAL_LIBRARIES ${CUDA_LIBRARIES})
    list(APPEND COLMAP_INTERNAL_LIBRARIES colmap_cuda)
endif()

if(COLMAP_GUI_ENABLED)
    find_package(Qt5 5.4 REQUIRED COMPONENTS Core OpenGL Widgets)
    list(APPEND COLMAP_EXTERNAL_LIBRARIES
        ${Qt5Core_LIBRARIES}
        ${Qt5OpenGL_LIBRARIES}
        ${Qt5Widgets_LIBRARIES})
    list(APPEND COLMAP_INCLUDE_DIRS
        ${Qt5Core_INCLUDE_DIRS}
        ${Qt5OpenGL_INCLUDE_DIRS}
        ${Qt5Widgets_INCLUDE_DIRS})
endif()

if(COLMAP_CGAL_ENABLED)
    find_package(CGAL REQUIRED)
    list(APPEND COLMAP_EXTERNAL_LIBRARIES ${CGAL_LIBRARY} ${GMP_LIBRARIES})
endif()

set(COLMAP_LIBRARIES
    colmap
    ${COLMAP_INTERNAL_LIBRARIES}
    ${COLMAP_EXTERNAL_LIBRARIES}
)

# Cleanup of configuration variables.

set(CMAKE_MODULE_PATH ${TEMP_CMAKE_MODULE_PATH})

unset(COLMAP_INSTALL_PREFIX)
unset(EIGEN3_INCLUDE_DIR_HINTS)
unset(FLANN_INCLUDE_DIR_HINTS)
unset(FLANN_LIBRARY_DIR_HINTS)
unset(LZ4_INCLUDE_DIR_HINTS)
unset(LZ4_LIBRARY_DIR_HINTS)
unset(FREEIMAGE_INCLUDE_DIR_HINTS)
unset(FREEIMAGE_LIBRARY_DIR_HINTS)
unset(METIS_INCLUDE_DIR_HINTS)
unset(METIS_LIBRARY_DIR_HINTS)
unset(GLEW_INCLUDE_DIR_HINTS)
unset(GLEW_LIBRARY_DIR_HINTS)
unset(GLOG_INCLUDE_DIR_HINTS)
unset(GLOG_LIBRARY_DIR_HINTS)
unset(SQLite3_INCLUDE_DIR_HINTS)
unset(SQLite3_LIBRARY_DIR_HINTS)
unset(QT5_CMAKE_CONFIG_DIR_HINTS)
