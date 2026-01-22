cmake_minimum_required(VERSION 3.21)
project(
  gpu_array
  VERSION 0.3.0
  LANGUAGES CXX)

# check if the project is top-level
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(GPU_ARRAY_IS_TOPLEVEL_PROJECT ON)
else()
  set(GPU_ARRAY_IS_TOPLEVEL_PROJECT OFF)
endif()

# options
option(ENABLE_HIP "Enable HIP support" OFF)
option(GPU_ARRAY_BUILD_TEST "Build ${PROJECT_NAME} tests"
       ${GPU_ARRAY_IS_TOPLEVEL_PROJECT})

# library
include(GNUInstallDirs)
add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)
target_compile_definitions(${PROJECT_NAME}
                           INTERFACE $<$<BOOL:${ENABLE_HIP}>:ENABLE_HIP>)
target_include_directories(
  ${PROJECT_NAME}
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION})

# install
if(GPU_ARRAY_IS_TOPLEVEL_PROJECT)
  install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}_Targets)

  install(
    EXPORT ${PROJECT_NAME}_Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
    ${PROJECT_NAME}ConfigVersion.cmake COMPATIBILITY SameMajorVersion
                                                     ARCH_INDEPENDENT)
  configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

  # install config files
  install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

  # install header files
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

# test
if(GPU_ARRAY_BUILD_TEST)
  enable_testing()
  add_subdirectory(test)
endif()
