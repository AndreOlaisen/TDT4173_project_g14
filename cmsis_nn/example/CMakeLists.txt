cmake_minimum_required(VERSION 3.13.1)

project("CMSIS-NN CIFAR10 example")

add_executable(example)
target_link_libraries(example CMSISNN)
target_sources(example PRIVATE
    main.c
    "${ROOT}/CMSIS/DSP/Source/SupportFunctions/arm_q7_to_float.c"
    "../util/nn_util.c")
target_include_directories(example PRIVATE "../util")
