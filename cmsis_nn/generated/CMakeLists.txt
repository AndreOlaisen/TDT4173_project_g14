cmake_minimum_required(VERSION 3.13.1)

project("Generated code example")

add_executable(generated)
target_link_libraries(generated CMSISNN -lpng)

set(cmsis_dep_sources
    "${ROOT}/CMSIS/DSP/Source/SupportFunctions/arm_q7_to_float.c")
FILE(GLOB app_sources ./*.c)
FILE(GLOB model_sources model/*.c)
target_sources(generated PRIVATE
               ${app_sources}
               ${model_sources}
               ${cmsis_dep_sources})
