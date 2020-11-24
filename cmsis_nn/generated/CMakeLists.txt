cmake_minimum_required(VERSION 3.13.1)

project("Generated code example")

add_executable(generated)
target_link_libraries(generated CMSISNN -lpng)

FILE(GLOB app_sources ./*.c)
FILE(GLOB model_sources model/*.c)
FILE(GLOB util_sources ../util/*.c)
target_sources(generated PRIVATE
               ${app_sources}
               ${model_sources}
               ${util_sources})
target_include_directories(generated PRIVATE ../util)
