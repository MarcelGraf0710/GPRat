# Inner CMakeLists.txt

### Options #######################################################################################
option(GPRAT_WITH_SYCL "Enable SYCL support with oneMath" OFF)

add_compile_definitions(
  GPRAT_WITH_CUDA=$<BOOL:${GPRAT_WITH_CUDA}>
  GPRAT_WITH_SYCL=$<BOOL:${GPRAT_WITH_SYCL}>
)

### Source files ##################################################################################

# Set general GPRAT source files
set(SOURCE_FILES
    src/gprat_c.cpp
    src/utils_c.cpp
    src/target.cpp
    src/gp_kernels.cpp
    src/gp_hyperparameters.cpp
    src/cpu/gp_functions.cpp
    src/cpu/gp_algorithms.cpp
    src/cpu/gp_uncertainty.cpp
    src/cpu/gp_optimizer.cpp
    src/cpu/tiled_algorithms.cpp
    src/cpu/adapter_cblas_fp32.cpp
    src/cpu/adapter_cblas_fp64.cpp)

# Add SYCL-specific source files if enabled
if(GPRAT_WITH_SYCL)
  list(
    APPEND
    SOURCE_FILES
    src/sycl/adapter_onemath.cpp
    src/sycl/sycl_gp_algorithms.cpp
    src/sycl/sycl_gp_functions.cpp
    src/sycl/sycl_gp_optimizer.cpp
    src/sycl/sycl_gp_uncertainty.cpp
    src/sycl/sycl_tiled_algorithms.cpp
    )
endif()
#src/sycl/sycl_kernels.cpp

message(STATUS "Added source files:")

foreach(source ${SOURCE_FILES})
  message("   ${source}")
endforeach()


### Add GPRat core library ########################################################################

# Add GPRAT core library
add_library(gprat_core STATIC ${SOURCE_FILES})

# We'd like to consume it just as core under the GPRat namespace
set_property(TARGET gprat_core PROPERTY EXPORT_NAME core)
add_library(GPRat::core ALIAS gprat_core)

### Headers #######################################################################################

# Add them as PRIVATE sources here so they show up in project files Can't use PUBLIC etc.
# see: https://stackoverflow.com/a/62465051

file(GLOB_RECURSE header_files CONFIGURE_DEPENDS include/*.hpp
     include/cpu/*.hpp)

if(GPRAT_WITH_SYCL)
  file(GLOB_RECURSE sycl_headers CONFIGURE_DEPENDS include/sycl/*.hpp)
  list(APPEND header_files ${sycl_headers})
endif()

message(STATUS "Added SYCL headers:")

foreach(header ${sycl_headers})
  message("   ${header}")
endforeach()

target_sources(gprat_core PRIVATE ${header_files})

### Linking #######################################################################################

# Link HPX libraries
target_link_libraries(gprat_core PUBLIC HPX::hpx)

# Link CUDA libaries if enabled
if(GPRAT_WITH_CUDA)
  target_link_libraries(gprat_core PUBLIC CUDA::cusolver CUDA::cublas)
endif()


# --- SYCL / oneMath linking --------------------------------------------------
if(GPRAT_WITH_SYCL)

  # simcl1
  list(APPEND CMAKE_PREFIX_PATH "/scratch-simcl1/grafml/Programs/oneMath/install/lib/cmake/oneMath")
  # set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda")
  message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
  message(STATUS "SYCL_FLAGS = ${SYCL_FLAGS}")

  # pcsgs11
  # list(APPEND CMAKE_PREFIX_PATH "/home/grafml/Programs/theRealOneMath/oneMath/install/lib/cmake/oneMath")

  message(STATUS "CMAKE_PREFIX_PATH = ${CMAKE_PREFIX_PATH}")
  find_package(oneMath REQUIRED)

  # target_compile_options(
  #   gprat_core PUBLIC
  #   -fsycl
  #   -fsycl-targets=nvptx64-nvidia-cuda
  #   -Xsycl-target-backend=nvptx64-nvidia-cuda
  #   --cuda-gpu-arch=sm_80
  # )

  set(SYCL_FLAGS 
    ${SYCL_FLAGS} 
    -fsycl 
    -fsycl-targets=nvptx64-nvidia-cuda 
    -Xsycl-target-backend=nvptx64-nvidia-cuda 
    --cuda-gpu-arch=sm_80
  )
  target_compile_options(gprat_core PUBLIC ${SYCL_FLAGS})

  # target_link_options(gprat_core PRIVATE -Xsycl-target-backend --fsycl-targets=sm_80)
  target_link_options(gprat_core PUBLIC ${SYCL_FLAGS})

  target_link_libraries(gprat_core PUBLIC
    ONEMATH::onemath_blas_cublas
    ONEMATH::onemath_lapack_cusolver
  )

  if (TARGET ONEMATH::onemath)
    target_link_libraries(gprat_core PUBLIC ONEMATH::onemath)
  endif()


# # Link SYCL libraries if enabled
# if(GPRAT_WITH_SYCL)
#   find_package(oneMath REQUIRED)
#   target_link_libraries(gprat_core PUBLIC CUDA::cusolver CUDA::cublas)
  
#   if (TARGET ONEMATH::onemath_blas_cublas)
#     target_link_libraries(dot_usm PRIVATE ONEMATH::onemath_blas_cublas)
#   else()
#   message(FATAL_ERROR "oneMath was not built with BLAS cuBLAS backend or target ONEMATH::onemath_blas_cublas not found")
#   endif()

#   if (TARGET ONEMATH::onemath_solver_cusolver)
#     target_link_libraries(dot_usm PRIVATE ONEMATH::onemath_solver_cusolver)
#   else()
#     message(FATAL_ERROR "oneMath was not built with SOLVER cuSOLVER backend or target ONEMATH::onemath_solver_cusolver not found")
#   endif()

#   # Also link the generic onemath (selector) if needed (some builds require it)
#   if (TARGET ONEMATH::onemath)
#     target_link_libraries(dot_usm PRIVATE ONEMATH::onemath)
#   endif()

endif()

# Include directories
target_include_directories(
  gprat_core PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/core/include>")

# Link BLAS
if(GPRAT_ENABLE_MKL)
  # Link Intel oneMKL
  target_link_libraries(gprat_core PUBLIC MKL::mkl_intel_lp64 MKL::mkl_core
                                          MKL::MKL MKL::mkl_sequential)
else()
  # Link OpenBLAS
  target_link_libraries(gprat_core PUBLIC ${OpenBLAS_LIB})
endif()

if(GPRAT_ENABLE_MKL)
  target_compile_definitions(gprat_core PUBLIC GPRAT_ENABLE_MKL)
endif()

target_compile_features(gprat_core PUBLIC cxx_std_17)

set_property(TARGET gprat_core PROPERTY POSITION_INDEPENDENT_CODE ON)

### Installation ##################################################################################

if(NOT CMAKE_SKIP_INSTALL_RULES)
  # We need to manually install those into CMAKE_INSTALL_INCLUDEDIR. Below
  # install(TARGETS ...) only setups the paths for the exported targets.
  install(
    DIRECTORY include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT Development)

  install(
    TARGETS gprat_core
    EXPORT GPRatTargets
    RUNTIME COMPONENT Runtime
    LIBRARY COMPONENT Runtime NAMELINK_COMPONENT Development
    ARCHIVE COMPONENT Development
    INCLUDES
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
endif()

### End of file ###################################################################################