cmake_minimum_required(VERSION 3.16)

project(gprat_cpp)

# Enables/disables computation with CUDA on test system
option(GPRAT_WITH_SYCL "Enable additional computation with SYCL, oneMath" OFF)

# This project can be built as part of GPRat itself or as a standalone project
# in which case below find_package() call is used.
if(TARGET GPRat::core)

  message("Using in-tree GPRat::core")

else()

  message("Using out-of-tree GPRat::core")

  find_package(GPRat REQUIRED)

  get_target_property(_gprat_defs GPRat::core INTERFACE_COMPILE_DEFINITIONS)
  get_target_property(_gprat_libs GPRat::core INTERFACE_LINK_LIBRARIES)

  foreach(def ${_gprat_defs})
    if(def MATCHES "GPRAT_WITH_SYCL")
      set(GPRAT_WITH_SYCL ON)
    endif()
  endforeach()

  foreach(lib ${_gprat_libs})
    if(lib MATCHES "ONEMATH")
      set(GPRAT_WITH_SYCL ON)
    endif()
  endforeach()

  if(GPRAT_WITH_SYCL)
    find_package(oneMath CONFIG REQUIRED)
  endif()
  
  if(USE_MKL)
    # Try to find Intel oneMKL
    set(MKL_INTERFACE_FULL "intel_lp64")
    set(MKL_THREADING "sequential")
    find_package(MKL CONFIG REQUIRED)
  endif()

endif()

# Add the executable
add_executable(gprat_cpp src/execute.cpp)

target_compile_features(gprat_cpp PUBLIC cxx_std_17)

# Link the libraries
target_link_libraries(gprat_cpp PUBLIC GPRat::core)
