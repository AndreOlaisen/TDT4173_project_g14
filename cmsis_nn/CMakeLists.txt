cmake_minimum_required(VERSION 3.13.1)
project(cmsis-nn)

# CMSIS-NN configuration
set(ROOT "${PROJECT_SOURCE_DIR}/CMSIS_5")
set(GCC ON)

add_subdirectory("${ROOT}/CMSIS/NN/Source")
add_subdirectory(example)
# add_subdirectory(generated)
