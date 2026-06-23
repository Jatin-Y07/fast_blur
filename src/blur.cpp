#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

static const int OMP_THRESHOLD = 200 * 200;
static const uint32_t FP_SHIFT = 20;
static const uint32_t FP_ROUND = 1u << (FP_SHIFT - 1); // 0.5 in fixed-point representation

// Horizontal pass splits the row into three zones to avoid branches and division in the main loop:
// 1: Growing window (x <= r)
// 2: Full window (interior) - uses fixed-point multiplication for the hot path
// 3: Shrinking window (x >= W - r)
static void blur_h(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, int W, int H, int C, int r) {
    const uint32_t full = (uint32_t)(2 * r + 1);
    const uint32_t MAGIC = ((1u << FP_SHIFT) + full / 2) / full;
    
    #pragma omp parallel for schedule(static) if(W * H > OMP_THRESHOLD)
    for (int y = 0; y < H; y++) {
        const uint8_t* row = src + (size_t)y * W * C;
        uint8_t* out = dst + (size_t)y * W * C;
        uint32_t s[4] = {}, cnt = 0;
        // Seed the window for x=0
        for (int k = 0; k <= r && k < W; k++) {
            for (int c = 0; c < C; c++) s[c] += row[k * C + c];
            cnt++;
        }  
        for (int c = 0; c < C; c++) out[c] = (uint8_t)((s[c] + cnt / 2) / cnt);
        int x = 1;
        // Zone 1: Window is still growing, adding new pixels
        for (; x <= r && x < W; x++) {
            if (x + r < W) { 
                for (int c = 0; c < C; c++) s[c] += row[(x + r) * C + c]; 
                cnt++; 
            }
            for (int c = 0; c < C; c++) out[x * C + c] = (uint8_t)((s[c] + cnt / 2) / cnt);
        }

        // Zone 2: Hot path (window size is constant). 
        // No division or branching here so the compiler can easily vectorize it.
        for (int z2end = W - r - 1; x <= z2end; x++) {
            for (int c = 0; c < C; c++) { 
                s[c] += row[(x + r) * C + c]; 
                s[c] -= row[(x - r - 1) * C + c]; 
            }
            for (int c = 0; c < C; c++) {
                out[x * C + c] = (uint8_t)((s[c] * MAGIC + FP_ROUND) >> FP_SHIFT);
            }
        }
        // Zone 3: Window is shrinking near the right edge, only removing pixels
        for (; x < W; x++) {
            if (x - r - 1 >= 0) { 
                for (int c = 0; c < C; c++) s[c] -= row[(x - r - 1) * C + c]; 
                cnt--; 
            }
            for (int c = 0; c < C; c++) out[x * C + c] = (uint8_t)((s[c] + cnt / 2) / cnt);
        }
    }
}
// Vertical pass using the exact same three-zone optimization
static void blur_v(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, int W, int H, int C, int r) {
    const uint32_t full = (uint32_t)(2 * r + 1);
    const uint32_t MAGIC = ((1u << FP_SHIFT) + full / 2) / full;
    #pragma omp parallel for schedule(static) if(W * H > OMP_THRESHOLD)
    for (int x = 0; x < W; x++) {
        uint32_t s[4] = {}, cnt = 0;
        for (int k = 0; k <= r && k < H; k++) {
            for (int c = 0; c < C; c++) s[c] += src[((size_t)k * W + x) * C + c];
            cnt++;
        }
        for (int c = 0; c < C; c++) dst[x * C + c] = (uint8_t)((s[c] + cnt / 2) / cnt);
        int y = 1;
        // Zone 1
        for (; y <= r && y < H; y++) {
            if (y + r < H) { 
                for (int c = 0; c < C; c++) s[c] += src[((size_t)(y + r) * W + x) * C + c]; 
                cnt++; 
            }
            for (int c = 0; c < C; c++) dst[((size_t)y * W + x) * C + c] = (uint8_t)((s[c] + cnt / 2) / cnt);
        }
        // Zone 2 (Hot path)
        for (int z2end = H - r - 1; y <= z2end; y++) {
            for (int c = 0; c < C; c++) {
                s[c] += src[((size_t)(y + r) * W + x) * C + c];
                s[c] -= src[((size_t)(y - r - 1) * W + x) * C + c];
            }
            for (int c = 0; c < C; c++) {
                dst[((size_t)y * W + x) * C + c] = (uint8_t)((s[c] * MAGIC + FP_ROUND) >> FP_SHIFT);
            }
        }
        // Zone 3
        for (; y < H; y++) {
            if (y - r - 1 >= 0) { 
                for (int c = 0; c < C; c++) s[c] -= src[((size_t)(y - r - 1) * W + x) * C + c]; 
                cnt--; 
            }
            for (int c = 0; c < C; c++) dst[((size_t)y * W + x) * C + c] = (uint8_t)((s[c] + cnt / 2) / cnt);
        }
    }
}
void box_blur_cpp(const uint8_t* input, uint8_t* output, int width, int height, int channels, float radius) {
    int r = (int)std::round(radius);
    // Using a uint8 temp buffer instead of floats so it plays nicely with the CPU cache
    std::vector<uint8_t> temp((size_t)width * height * channels);
    blur_h(input, temp.data(), width, height, channels, r);
    blur_v(temp.data(), output, width, height, channels, r);
}
py::array_t<uint8_t> blur(py::array_t<uint8_t> input, float radius) {
    auto buf = input.request();
    if (buf.ndim != 3) {
        throw std::runtime_error("Expected an HxWxC array");}
    int H = (int)buf.shape[0];
    int W = (int)buf.shape[1];
    int C = (int)buf.shape[2];
    auto output = py::array_t<uint8_t>((size_t)H * W * C);
    auto out_buf = output.request();
    {
        // Release GIL while doing the heavy C++ OpenMP work
        py::gil_scoped_release release;
        box_blur_cpp(static_cast<const uint8_t*>(buf.ptr),
                     static_cast<uint8_t*>(out_buf.ptr), W, H, C, radius);
    }   
    output.resize({H, W, C});
    return output;
}
PYBIND11_MODULE(fast_blur, m) {
    m.doc() = "Optimized separable box blur utilizing a fixed-point hot path and OpenMP";
    m.def("blur", &blur, py::arg("image"), py::arg("radius") = 5.0f,
          "Blurs an HxWxC uint8 array. Radius accepts floats (matches PIL behavior).");
}