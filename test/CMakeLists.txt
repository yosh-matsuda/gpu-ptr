cmake_minimum_required(VERSION 3.21)

#
# COMPILER/LINKER FLAGS
#
if(UNIX)
  # Warnings
  set(ADD_CXX_FLAGS
      "-Wall -Wextra -Wcast-align -Wcast-qual -Wdelete-non-virtual-dtor -Woverloaded-virtual -Wpointer-arith -Winit-self -Wredundant-decls -Wsign-promo -Wswitch-default -Wswitch-enum -Wvariadic-macros -Wwrite-strings -Wno-unknown-pragmas"
  )

  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(ADD_CXX_FLAGS "${ADD_CXX_FLAGS} -Wshadow")
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    set(ADD_CXX_FLAGS "${ADD_CXX_FLAGS} -Wshadow=local -Wno-pragmas")
  endif()
else() # WIN32
  set(ADD_CXX_FLAGS "${ADD_CXX_FLAGS} /W4")
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

#
# CUDA/HIP support
#
include(CheckLanguage)
if(ENABLE_HIP)
  # determine HIP platform
  if(NOT CMAKE_HIP_PLATFORM AND NOT $ENV{HIP_PLATFORM})
    set(CMAKE_HIP_PLATFORM $ENV{HIP_PLATFORM})
  endif()

  # check HIP support
  check_language(HIP)
  if(NOT CMAKE_HIP_COMPILER)
    message(FATAL_ERROR "No HIP supported")
  endif()

  set(GPU_LANGUAGE "HIP")
  if(CMAKE_HIP_PLATFORM STREQUAL "nvidia")
    set(GPU_PLATFORM "NVIDIA")
  else()
    set(GPU_PLATFORM "AMD")
  endif()

  set(CMAKE_HIP_STANDARD 20)
  set(CMAKE_HIP_EXTENSIONS OFF)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)

  if(NOT CMAKE_HIP_ARCHITECTURES)
    if(GPU_PLATFORM STREQUAL "NVIDIA")
      set(CMAKE_HIP_ARCHITECTURES 70 80 90)
      set(ADD_CXX_FLAGS "${ADD_CXX_FLAGS} -Xptxas=-v -use_fast_math")
      set(CMAKE_HIP_FLAGS_DEBUG "${CMAKE_HIP_FLAGS_DEBUG}")
    else()
      set(ADD_CXX_FLAGS
          "${ADD_CXX_FLAGS} -Rpass-analysis=kernel-resource-usage -ffast-math")
      set(CMAKE_HIP_ARCHITECTURES "gfx942")
    endif()
  endif()

  enable_language(HIP)
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} ${ADD_CXX_FLAGS}")
  set(CMAKE_HIP_FLAGS_DEBUG "${CMAKE_HIP_FLAGS_DEBUG} ${ADD_CXX_FLAGS}")
  set(CMAKE_HIP_FLAGS_RELEASE "${CMAKE_HIP_FLAGS_RELEASE} ${ADD_CXX_FLAGS}")
  set(CMAKE_HIP_FLAGS_RELWITHDEBINFO
      "${CMAKE_HIP_FLAGS_RELWITHDEBINFO} ${ADD_CXX_FLAGS}")
else()
  # check CUDA support
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "No CUDA supported")
  endif()

  set(GPU_LANGUAGE "CUDA")
  set(GPU_PLATFORM "NVIDIA")
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 90)
  endif()

  enable_language(CUDA)
  set(ADD_CXX_FLAGS
      "${ADD_CXX_FLAGS} -Xptxas=-v -use_fast_math --expt-relaxed-constexpr")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${ADD_CXX_FLAGS}")
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} ${ADD_CXX_FLAGS}")
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} ${ADD_CXX_FLAGS}")
  set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO
      "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} ${ADD_CXX_FLAGS}")
endif()

# MESSAGE
message(STATUS "ENABLE_HIP: ${ENABLE_HIP}")
message(STATUS "GPU_PLATFORM: ${GPU_PLATFORM}")
message(
  STATUS
    "CMAKE_CUDA/HIP_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}${CMAKE_HIP_ARCHITECTURES}"
)

# GTest
find_package(GTest REQUIRED)

# executables
add_executable("${PROJECT_NAME}_test" test.cpp)
set_source_files_properties(test.cpp PROPERTIES LANGUAGE ${GPU_LANGUAGE})
target_link_libraries("${PROJECT_NAME}_test"
                      PRIVATE ${PROJECT_NAME} GTest::gtest GTest::gtest_main)

# CUDA Toolkit as system header
if(GPU_PLATFORM STREQUAL NVIDIA)
  target_include_directories(
    "${PROJECT_NAME}_test" SYSTEM
    PUBLIC ${CMAKE_HIP_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

# add google test
include(GoogleTest)
gtest_discover_tests("${PROJECT_NAME}_test" DISCOVERY_MODE PRE_TEST)

# compile commands
set(ENV{CMAKE_EXPORT_COMPILE_COMMANDS} ON)
if(CMAKE_EXPORT_COMPILE_COMMANDS)
  add_custom_target(
    gen_compile_commands ALL
    COMMAND ${CMAKE_COMMAND} -E touch
            ${CMAKE_BINARY_DIR}/compile_commands.json # trick to always run
    DEPENDS ${CMAKE_SOURCE_DIR}/compile_commands.json)

  # Generate a compilation database with header files
  if(NOT COMPDB)
    find_program(COMPDB compdb)
  endif()
  if(COMPDB)
    # Run compdb
    add_custom_command(
      OUTPUT ${CMAKE_SOURCE_DIR}/compile_commands.json
      DEPENDS ${CMAKE_BINARY_DIR}/compile_commands.json
      COMMAND ${CMAKE_COMMAND} -E remove
              ${CMAKE_SOURCE_DIR}/compile_commands.json
      COMMAND ${COMPDB} list > ${CMAKE_SOURCE_DIR}/compile_commands.json)
  else()
    message(AUTHOR_WARNING "'compdb' not found")

    # Copy file instead of compdb
    add_custom_command(
      OUTPUT ${CMAKE_SOURCE_DIR}/compile_commands.json
      DEPENDS ${CMAKE_BINARY_DIR}/compile_commands.json
      COMMAND
        ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/compile_commands.json
        ${CMAKE_SOURCE_DIR}/compile_commands.json)
  endif()

  # Tell clangd to define __NVCC__ or __HIP__ for CUDA or HIP files
  if(NOT SED)
    find_program(SED sed)
  endif()
  if(SED)
    add_custom_command(
      OUTPUT ${CMAKE_SOURCE_DIR}/compile_commands.json
      APPEND
      COMMAND
        ${SED} -i -e 's|nvcc|nvcc -D__NVCC__|g' -e
        's|-x hip|-x hip -D__HIP__|g' ${CMAKE_SOURCE_DIR}/compile_commands.json)
  else()
    message(AUTHOR_WARNING "'sed' not found")
  endif()

endif()
