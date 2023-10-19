cmake_minimum_required(VERSION 3.13)

project(self_organizing_gmm
	VERSION 0.0.0
	DESCRIPTION
	"PRI Mapping"
	LANGUAGES CXX C)

# configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
set(CMAKE_CXX_FLAGS "-std=c++17 -stdlib=libc++ -march=native")
else()
set(CMAKE_CXX_FLAGS "-std=c++17 -march=native -fopenmp -lstdc++fs")
endif()
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

set(FILE_SYS_LIB "stdc++fs")

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp /usr/local/lib/libomp.dylib -I/usr/local/include")
  set(OpenMP_CXX_LIB_NAMES "libomp")
  set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp /usr/local/lib/libomp.dylib -I/usr/local/include")
  set(OpenMP_C_LIB_NAMES="libomp")
  set(OpenMP_libomp_LIBRARY "/usr/local/lib/libomp.dylib")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(FILE_SYS_LIB "")
endif()

# find dependencies
find_package(OpenMP REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Eigen3 3.4 REQUIRED)
find_package(GSL REQUIRED)
find_package(nanoflann REQUIRED)

# Setup pybind modules
pybind11_add_module(time_profiler_py src/pybind/time_profiler_py.cpp)
target_include_directories(time_profiler_py
  PRIVATE
  include
)
target_link_libraries(
  time_profiler_py
  PRIVATE stdc++fs
)
set_target_properties(time_profiler_py PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX})

pybind11_add_module(mean_shift_py src/pybind/mean_shift_py.cpp)
target_include_directories(mean_shift_py PRIVATE include)
target_link_libraries(
  mean_shift_py
  PRIVATE ${FILE_SYS_LIB}
  PRIVATE Eigen3::Eigen
  PRIVATE nanoflann::nanoflann
)
set_target_properties(mean_shift_py PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX})

pybind11_add_module(camera_model_py src/pybind/camera_model_py.cpp)
target_include_directories(camera_model_py PRIVATE include)
target_link_libraries(
  camera_model_py
  PRIVATE ${FILE_SYS_LIB}
  PRIVATE Eigen3::Eigen
)
set_target_properties(camera_model_py PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX})

pybind11_add_module(gmm_py src/pybind/gmm_py.cpp)
target_include_directories(gmm_py PRIVATE include ${GSL_INCLUDE_DIRS})
target_link_libraries(
  gmm_py
  PRIVATE ${FILE_SYS_LIB}
  PRIVATE Eigen3::Eigen
  PRIVATE ${GSL_LIBRARIES}
  PRIVATE ${OpenMP_libomp_LIBRARY}
)
set_target_properties(gmm_py PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX})

pybind11_add_module(kinit_py src/pybind/kinit_py.cpp)
target_include_directories(kinit_py PRIVATE include)
target_link_libraries(
  kinit_py
  PRIVATE ${FILE_SYS_LIB}
  PRIVATE Eigen3::Eigen
  PRIVATE ${OpenMP_libomp_LIBRARY}
)
set_target_properties(kinit_py PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX})

pybind11_add_module(sogmm_cpu src/pybind/sogmm_cpu.cpp)
target_include_directories(sogmm_cpu PRIVATE include ${GSL_INCLUDE_DIRS})
target_link_libraries(
  sogmm_cpu
  PRIVATE ${FILE_SYS_LIB}
  PRIVATE Eigen3::Eigen
  PRIVATE ${GSL_LIBRARIES}
  PRIVATE ${OpenMP_libomp_LIBRARY}
  PRIVATE nanoflann::nanoflann
)
set_target_properties(sogmm_cpu PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX})

# Install
install(TARGETS DESTINATION ${CMAKE_INSTALL_PREFIX})

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION "include/${PROJECT_NAME}"
  FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY cmake/
  DESTINATION "share/${PROJECT_NAME}"
  FILES_MATCHING PATTERN "*.cmake"
)
