cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(cudamat_cmake LANGUAGES C CXX CUDA)

add_subdirectory(cuda)
add_subdirectory(python)