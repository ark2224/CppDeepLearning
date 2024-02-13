# cmake_minimum_required(VERSION 3.10)
# project(CppDeepLearning VERSION 1.0)

# list(APPEND CMAKE_PREFIX_PATH "<libtorch-path>")
# find_package(Torch REQUIRED)

# include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# add_executable(CppDeepLearning main.cpp FeedForwardNN/src/ffnn.cpp FeedForwardNN/src/init.cpp)
# target_link_libraries(CppDeepLearning "${TORCH_LIBRARIES}")
# set_property(TARGET CppDeepLearning PROPERTY CXX_STANDARD 17)
# # including header files below in the /include directory:
# target_include_directories(CppDeepLearning PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/FeedForwardNN/include)


# # The following code block is suggested to be used on Windows.
# # According to https://github.com/pytorch/pytorch/issues/25457,
# # the DLLs need to be copied to avoid memory errors.
# if (MSVC)
#   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#   add_custom_command(TARGET CppDeepLearning
#                      POST_BUILD
#                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                      ${TORCH_DLLS}
#                      $<TARGET_FILE_DIR:CppDeepLearning>)
# endif (MSVC)

cmake_minimum_required(VERSION 3.5)
project(CppDeepLearning)
set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)

option(DOWNLOAD_MNIST "Download the MNIST dataset from the internet" ON)
if (DOWNLOAD_MNIST)
  message(STATUS "Downloading MNIST dataset")
  execute_process(
    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/tools/download_mnist.py
      -d ${CMAKE_BINARY_DIR}/data
        # -d ./data
    ERROR_VARIABLE DOWNLOAD_ERROR)
  if (DOWNLOAD_ERROR)
    message(FATAL_ERROR "Error downloading MNIST dataset: ${DOWNLOAD_ERROR}")
  endif()
endif()

add_executable(CppDeepLearning main.cpp FeedForwardNN/src/ffnn.cpp FeedForwardNN/src/init.cpp)
target_compile_features(CppDeepLearning PUBLIC cxx_range_for)
target_link_libraries(CppDeepLearning ${TORCH_LIBRARIES})

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET CppDeepLearning
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:CppDeepLearning>)
endif (MSVC)