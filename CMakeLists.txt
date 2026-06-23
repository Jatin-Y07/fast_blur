cmake_minimum_required(VERSION 3.15)
project(fast_blur)

# Auto-find pybind11 from pip installation
execute_process(
  COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 REQUIRED)

pybind11_add_module(fast_blur src/blur.cpp)

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
