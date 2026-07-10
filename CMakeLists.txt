cmake_minimum_required(VERSION 3.24)

project(ckks_obs_ctrl LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CHEDDAR_DIR "$ENV{HOME}/Cheddar_project/cheddar-fhe")

set(BUILD_UNITTEST OFF CACHE BOOL "" FORCE)
set(ENABLE_EXTENSION ON CACHE BOOL "" FORCE)
set(USE_GMP ON CACHE BOOL "" FORCE)

add_subdirectory(${CHEDDAR_DIR} ${CMAKE_BINARY_DIR}/cheddar-build)

find_package(nlohmann_json CONFIG REQUIRED)

add_executable(ckks_sanity
    ckks_sanity.cpp
)

target_include_directories(ckks_sanity PRIVATE
    .
    ${CHEDDAR_DIR}/include
)

target_link_libraries(ckks_sanity PRIVATE
    cheddar
    nlohmann_json::nlohmann_json
)

add_executable(ckks_obs_standalone
    ckks_obs_standalone.cpp
)

target_include_directories(ckks_obs_standalone PRIVATE
    .
    ${CHEDDAR_DIR}/include
)

target_link_libraries(ckks_obs_standalone PRIVATE
    cheddar
    nlohmann_json::nlohmann_json
)

add_executable(ckks_closedloop_200
    ckks_closedloop_200.cpp
)

target_include_directories(ckks_closedloop_200 PRIVATE
    .
    ${CHEDDAR_DIR}/include
)

target_link_libraries(ckks_closedloop_200 PRIVATE
    cheddar
    nlohmann_json::nlohmann_json
)

add_executable(ckks_boot_sanity
    ckks_boot_sanity.cpp
)

target_include_directories(ckks_boot_sanity PRIVATE
    .
    ${CHEDDAR_DIR}/include
)

target_link_libraries(ckks_boot_sanity PRIVATE
    cheddar
    nlohmann_json::nlohmann_json
)
add_executable(ckks_closedloop_200_boot
    ckks_closedloop_200_boot.cpp
)

target_include_directories(ckks_closedloop_200_boot PRIVATE
    .
    ${CHEDDAR_DIR}/include
)

target_link_libraries(ckks_closedloop_200_boot PRIVATE
    cheddar
    nlohmann_json::nlohmann_json
)


add_executable(ckks_closedloop_200_boot_packed
    ckks_closedloop_200_boot_packed.cpp
)

target_include_directories(ckks_closedloop_200_boot_packed PRIVATE
    .
    ${CHEDDAR_DIR}/include
)

target_link_libraries(ckks_closedloop_200_boot_packed PRIVATE
    cheddar
    nlohmann_json::nlohmann_json
)

