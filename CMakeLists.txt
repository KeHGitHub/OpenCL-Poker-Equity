cmake_minimum_required(VERSION 3.16)

project(OpenCLPokerEquity LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCL REQUIRED)

add_executable(poker_equity
    src/main.cpp
)

target_link_libraries(poker_equity PRIVATE OpenCL::OpenCL)

target_compile_definitions(poker_equity PRIVATE CL_TARGET_OPENCL_VERSION=120)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/equity_kernel.cl
    ${CMAKE_CURRENT_BINARY_DIR}/kernels/equity_kernel.cl
    COPYONLY
)
