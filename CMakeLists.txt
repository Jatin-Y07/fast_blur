cmake_minimum_required(VERSION 3.15)
project(fast_blur)

find_package(pybind11 REQUIRED)

pybind11_add_module(fast_blur src/blur.cpp)

# Mac-specific OpenMP setup (Homebrew libomp)
if(APPLE)
    target_compile_options(fast_blur PRIVATE
        -O3 -march=native -ffast-math
        -Xpreprocessor -fopenmp
        -I/opt/homebrew/opt/libomp/include
    )
    target_link_options(fast_blur PRIVATE
        -L/opt/homebrew/opt/libomp/lib -lomp
    )
else()
    target_compile_options(fast_blur PRIVATE -O3 -march=native -ffast-math -fopenmp)
    target_link_options(fast_blur PRIVATE -fopenmp)
endif()
