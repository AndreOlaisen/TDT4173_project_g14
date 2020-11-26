#
# Copyright (c) 2019 Nordic Semiconductor
#
# SPDX-License-Identifier: LicenseRef-BSD-5-Clause-Nordic
#

cmake_minimum_required(VERSION 3.13.1)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project("CMSIS-NN sample")

set(app_sources src/main.c)

if (NOT CONFIG_BOARD_NRF9160DK_NRF52840)
FILE(GLOB model_sources model/*.c)
set (app_sources ${app_sources} ${model_sources})
endif()

target_sources(app PRIVATE ${app_sources})
target_include_directories(app PRIVATE .)
