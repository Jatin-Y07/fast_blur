#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

static const int      OMP_THRESHOLD = 200 * 200;
static const uint32_t FP_SHIFT = 20;
static const uint32_t FP_ROUND = 1u << (FP_SHIFT - 1);   // 0.5 in fixed-point

// ─── HORIZONTAL PASS ────────────────────────────────────────────────────────
// Three zones per row — Zone 2 (interior) has NO division, NO branches:
//   Zone 1  x ∈ [0,   r]       window growing   → integer divide (only 2r pixels)
//   Zone 2  x ∈ [r+1, W-r-1]   window = 2r+1    → fixed-point multiply (hot path)
//   Zone 3  x ∈ [W-r, W-1]     window shrinking → integer divide (only 2r pixels)
static void blur_h(const uint8_t* __restrict__ src,
                   uint8_t* __restrict__       dst,
                   int W, int H, int C, int r) {
    const uint32_t full  = (uint32_t)(2*r + 1);
    const uint32_t MAGIC = ((1u << FP_SHIFT) + full/2) / full; // round(2^20 / full)

    #pragma omp parallel for schedule(static) if(W * H > OMP_THRESHOLD)
    for (int y = 0; y < H; y++) {
        const uint8_t* row = src + (size_t)y * W * C;
        uint8_t*       out = dst + (size_t)y * W * C;
        uint32_t s[4] = {}, cnt = 0;

        // Seed window for x=0: pixels [0, min(r, W-1)]
        for (int k = 0; k <= r && k < W; k++) {
            for (int c = 0; c < C; c++) s[c] += row[k*C+c];
            cnt++;
        }
        // Output x=0 (Zone 1 boundary)
        for (int c=0;c<C;c++) out[c] = (uint8_t)((s[c] + cnt/2) / cnt);

        int x = 1;

        // Zone 1: window still growing, only additions
        for (; x <= r && x < W; x++) {
            if (x+r < W) { for (int c=0;c<C;c++) s[c] += row[(x+r)*C+c]; cnt++; }
            for (int c=0;c<C;c++) out[x*C+c] = (uint8_t)((s[c]+cnt/2)/cnt);
        }
        // cnt == full = 2r+1 here

        // Zone 2: hot path — no division, no branches, compiler can vectorize
        for (int z2end = W-r-1; x <= z2end; x++) {
            for (int c=0;c<C;c++) { s[c] += row[(x+r)*C+c]; s[c] -= row[(x-r-1)*C+c]; }
            for (int c=0;c<C;c++) out[x*C+c] = (uint8_t)((s[c]*MAGIC + FP_ROUND) >> FP_SHIFT);
        }

        // Zone 3: window shrinking, only removals
        for (; x < W; x++) {
            if (x-r-1 >= 0) { for (int c=0;c<C;c++) s[c] -= row[(x-r-1)*C+c]; cnt--; }
            for (int c=0;c<C;c++) out[x*C+c] = (uint8_t)((s[c]+cnt/2)/cnt);
        }
    }
}

// ─── VERTICAL PASS ──────────────────────────────────────────────────────────
// Same three-zone trick — eliminates division in the interior (hot path)
static void blur_v(const uint8_t* __restrict__ src,
                   uint8_t* __restrict__       dst,
                   int W, int H, int C, int r) {
    const uint32_t full  = (uint32_t)(2*r + 1);
    const uint32_t MAGIC = ((1u << FP_SHIFT) + full/2) / full;

    #pragma omp parallel for schedule(static) if(W * H > OMP_THRESHOLD)
    for (int x = 0; x < W; x++) {
        uint32_t s[4] = {}, cnt = 0;

        for (int k = 0; k <= r && k < H; k++) {
            for (int c=0;c<C;c++) s[c] += src[((size_t)k*W+x)*C+c];
            cnt++;
        }
        for (int c=0;c<C;c++) dst[x*C+c] = (uint8_t)((s[c]+cnt/2)/cnt);

        int y = 1;

        // Zone 1
        for (; y <= r && y < H; y++) {
            if (y+r < H) { for (int c=0;c<C;c++) s[c] += src[((size_t)(y+r)*W+x)*C+c]; cnt++; }
            for (int c=0;c<C;c++) dst[((size_t)y*W+x)*C+c] = (uint8_t)((s[c]+cnt/2)/cnt);
        }

        // Zone 2: hot path
        for (int z2end = H-r-1; y <= z2end; y++) {
            for (int c=0;c<C;c++) {
                s[c] += src[((size_t)(y+r  )*W+x)*C+c];
                s[c] -= src[((size_t)(y-r-1)*W+x)*C+c];
            }
            for (int c=0;c<C;c++)
                dst[((size_t)y*W+x)*C+c] = (uint8_t)((s[c]*MAGIC + FP_ROUND) >> FP_SHIFT);
        }

        // Zone 3
        for (; y < H; y++) {
            if (y-r-1 >= 0) { for (int c=0;c<C;c++) s[c] -= src[((size_t)(y-r-1)*W+x)*C+c]; cnt--; }
            for (int c=0;c<C;c++) dst[((size_t)y*W+x)*C+c] = (uint8_t)((s[c]+cnt/2)/cnt);
        }
    }
}

// ─── ENTRY POINT ────────────────────────────────────────────────────────────
void box_blur_cpp(const uint8_t* input, uint8_t* output,
                  int width, int height, int channels, float radius) {
    int r = (int)std::round(radius);
    // uint8 temp = 4x smaller than float → fits in L1/L2 cache much better
    std::vector<uint8_t> temp((size_t)width * height * channels);
    blur_h(input,       temp.data(), width, height, channels, r);
    blur_v(temp.data(), output,      width, height, channels, r);
}

// ─── PYBIND11 WRAPPER ───────────────────────────────────────────────────────
py::array_t<uint8_t> blur(py::array_t<uint8_t> input, float radius) {
    auto buf = input.request();
    if (buf.ndim != 3) throw std::runtime_error("Expected H×W×C array");
    int H = (int)buf.shape[0], W = (int)buf.shape[1], C = (int)buf.shape[2];

    auto output  = py::array_t<uint8_t>((size_t)H * W * C);
    auto out_buf = output.request();
    {
        py::gil_scoped_release release;
        box_blur_cpp(static_cast<const uint8_t*>(buf.ptr),
                     static_cast<uint8_t*>(out_buf.ptr), W, H, C, radius);
    }
    output.resize({H, W, C});
    return output;
}

PYBIND11_MODULE(fast_blur, m) {
    m.doc() = "Three-zone separable box blur: fixed-point hot path, uint8 temp, OpenMP";
    m.def("blur", &blur, py::arg("image"), py::arg("radius") = 5.0f,
          "Blur H×W×C uint8 array. Radius accepts floats like PIL.ImageFilter.BoxBlur.");
}