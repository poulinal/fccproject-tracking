cmake_minimum_required(VERSION 3.10)
project(dch_python_module)

set(ENV{TMPDIR} "/eos/user/a/alpoulin/tmp")

# message(STATUS "ENV{TMPDIR}: $ENV{TMPDIR}")
# message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
# message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
# message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")


# Find required packages
find_package(ROOT REQUIRED)
set(DD4hep_DIR "/cvmfs/sw.hsf.org/key4hep/releases/2024-10-03/x86_64-almalinux9-gcc14.2.0-opt/dd4hep/1.30-fx72h5/cmake")
# set(DD4hep_DIR "/cvmfs/sw.hsf.org/key4hep/releases/2024-10-03/x86_64-almalinux9-gcc14.2.0-opt/dd4hep/1.30-fx72h5/lib")
find_package(DD4hep REQUIRED COMPONENTS DDRec)
find_package(pybind11 REQUIRED)

# Find ROOT with all components
find_package(ROOT REQUIRED COMPONENTS Core Geom MathCore Physics)

# Enable ROOT's CMake macros
include(${ROOT_USE_FILE})

# Include directories
include_directories(${ROOT_INCLUDE_DIRS})
include_directories(${DD4hep_INCLUDE_DIRS})


# include_directories(/cvmfs/sw.hsf.org/key4hep/releases/2024-10-03/x86_64-almalinux9-gcc14.2.0-opt/dd4hep/1.30-fx72h5/include/DDRec/DCH_info.h)
# include_directories(/cvmfs/sw.hsf.org/key4hep/releases/2024-10-03/x86_64-almalinux9-gcc14.2.0-opt/dd4hep/1.30-fx72h5/include)
# include_directories(/cvmfs/sw.hsf.org/key4hep/releases/2024-10-03/x86_64-almalinux9-gcc14.2.0-opt/dd4hep/1.30-fx72h5/lib)
# include_directories(/cvmfs/sw.hsf.org/key4hep/releases/2024-10-03/x86_64-almalinux9-gcc14.2.0-opt/dd4hep/1.30-fx72h5/lib)
link_directories(/cvmfs/sw.hsf.org/key4hep/releases/2024-10-03/x86_64-almalinux9-gcc14.2.0-opt/dd4hep/1.30-fx72h5/lib)
link_directories(/cvmfs/sw.hsf.org/key4hep/releases/2024-10-03/x86_64-almalinux9-gcc14.2.0-dbg/dd4hep/1.30-37zydz/lib/)
link_directories(/cvmfs/sw.hsf.org/key4hep/releases/2024-10-03/x86_64-almalinux9-gcc14.2.0-opt/dd4hep/1.30-fx72h5/cmake)

message(STATUS "DD4hep_LIBRARIES: ${DD4hep_LIBRARIES}")
message(STATUS "DD4hep_LIBRARIES_DIRS: ${DD4hep_LIBRARIES_DIRS}")
message(STATUS "DD4hep_INCLUDE_DIRS: ${DD4hep_INCLUDE_DIRS}")

# find_library(DDRec_LIBRARY NAMES DDRec PATHS ${DD4hep_LIBRARY_DIRS} NO_DEFAULT_PATH)
find_library(DDRec_LIBRARY NAMES DDRec PATHS /cvmfs/sw.hsf.org/key4hep/releases/2024-10-03/x86_64-almalinux9-gcc14.2.0-opt/dd4hep/1.30-fx72h5/lib NO_DEFAULT_PATH)
find_library(DDRec_INCLUDE NAMES DDRec PATHS ${DD4hep_INCLUDE_DIRS} NO_DEFAULT_PATH)

if(DDRec_LIBRARY)
    message(STATUS "Found DDRec library: ${DDRec_LIBRARY}")
else()
    if (DDRec_INCLUDE)
        message(STATUS "Found DDRec include: ${DDRec_INCLUDE}")
    else()
        message(FATAL_ERROR "DDRec library not found in ${DD4hep_INCLUDE_DIRS} and ${DD4hep_LIBRARY_DIRS}")
    endif()
    message(FATAL_ERROR "DDRec library not found in ${DD4hep_LIBRARY_DIRS}")
endif()



# # Try to find the DDRec library explicitly
# find_library(DDRec_LIB NAMES DDRec 
#              PATHS ${DD4hep_LIBRARY_DIRS} 
#              DOC "Path to DDRec library")
# message(STATUS "Found DDRec library: ${DDRec_LIB}")


set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-strip")
# set(CMAKE_SHARED_LINKER_FLAGS "")

# Add the pybind11 module
pybind11_add_module(dch_module dch_python_bindings.cpp)

# Link libraries
target_link_libraries(dch_module PRIVATE ${ROOT_LIBRARIES} ${DD4hep_LIBRARIES} ${DDRec_LIBRARY} /cvmfs/sw.hsf.org/key4hep/releases/2024-10-03/x86_64-almalinux9-gcc14.2.0-opt/dd4hep/1.30-fx72h5/lib/libDDRec.so /cvmfs/sw.hsf.org/key4hep/releases/2024-10-03/x86_64-almalinux9-gcc14.2.0-dbg/dd4hep/1.30-37zydz/lib/libDDRec.so)
# target_link_libraries(dch_module PRIVATE ${ROOT_LIBRARIES} ${DD4hep_LIBRARIES} DDRec /cvmfs/sw.hsf.org/key4hep/releases/2024-10-03/x86_64-almalinux9-gcc14.2.0-opt/dd4hep/1.30-fx72h5/lib/libDDRec.so)

# /cvmfs/sw.hsf.org/key4hep/releases/2024-10-03/x86_64-almalinux9-gcc14.2.0-dbg/dd4hep/1.30-37zydz/lib/libDDRec.so
