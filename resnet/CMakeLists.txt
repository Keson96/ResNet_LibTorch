cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(resnet)

find_package(Torch REQUIRED)

add_executable(resnet resnet.cpp)
target_link_libraries(resnet "${TORCH_LIBRARIES}")
set_property(TARGET resnet PROPERTY CXX_STANDARD 11)
