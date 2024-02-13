cmake_minimum_required(VERSION 3.10)
project(CppDeepLearning VERSION 1.0)

list(APPEND CMAKE_PREFIX_PATH "<libtorch-path>")
find_package(Torch REQUIRED)
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(CppDeepLearning main.cpp)
target_link_libraries(CppDeepLearning "${TORCH_LIBRARIES}")
set_property(TARGET CppDeepLearning PROPERTY CXX_STANDARD 17)
# including header files below in the /include directory:
# target_include_directories(CppDeepLearning PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)


# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET CppDeepLearning
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:CppDeepLearning>)
endif (MSVC)