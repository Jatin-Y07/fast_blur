import sys, time, os, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "build"))

import numpy as np
import matplotlib.pyplot as plt
import fast_blur
from python_impl.blur import box_blur
from PIL import Image, ImageFilter

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Benchmark fast_blur vs Python vs PIL")
parser.add_argument("--image", type=str, default=None,
                    help="Path to a real image (optional).")
parser.add_argument("--radius", type=float, default=3.0,
                    help="Blur radius, supports floats like PIL (default: 3.0)")
parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512, 1024, 2048],
                    help="Square sizes to benchmark (default:  512 1024 2048)")
args = parser.parse_args()

RADIUS = args.radius

# ---------------------------------------------------------------------------
# Image loader: real photo or synthetic
# ---------------------------------------------------------------------------
def get_image(size):
    if args.image and os.path.exists(args.image):
        pil_img = Image.open(args.image).convert("RGB").resize((size, size))
        return np.array(pil_img, dtype=np.uint8)
    return np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)

# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------
sizes = args.sizes
py_times, pil_times, cpp_times = [], [], []

print(f"\nRadius: {RADIUS}  |  Source: {args.image or 'synthetic'}\n")
print(f"{'Size':<10} {'Python':>10} {'PIL':>10} {'C++':>10} {'vs Python':>12} {'vs PIL':>10}")
print("-" * 68)

for s in sizes:
    img_np = get_image(s)
    img_pil = Image.fromarray(img_np)
    flat = img_np.flatten().tobytes()

    # --- Pure Python (1 run) ---
    start = time.perf_counter()
    box_blur(flat, s, s, 3, int(RADIUS))
    py_t = time.perf_counter() - start

    # --- PIL BoxBlur (3 runs) ---
    pil_runs = []
    for _ in range(3):
        start = time.perf_counter()
        img_pil.filter(ImageFilter.BoxBlur(RADIUS))
        pil_runs.append(time.perf_counter() - start)
    pil_t = sum(pil_runs) / len(pil_runs)

    # --- C++ pybind11 (3 runs) ---
    cpp_runs = []
    for _ in range(3):
        start = time.perf_counter()
        fast_blur.blur(img_np, radius=RADIUS)
        cpp_runs.append(time.perf_counter() - start)
    cpp_t = sum(cpp_runs) / len(cpp_runs)

    py_times.append(py_t)
    pil_times.append(pil_t)
    cpp_times.append(cpp_t)

    vs_py  = py_t  / cpp_t
    vs_pil = pil_t / cpp_t
    print(f"{s}x{s:<7} {py_t:>9.3f}s {pil_t:>9.4f}s {cpp_t:>9.4f}s "
          f"{vs_py:>10.1f}x {vs_pil:>9.2f}x")

# ---------------------------------------------------------------------------
# Save blurred sample if a real image was provided
# ---------------------------------------------------------------------------
if args.image and os.path.exists(args.image):
    orig = Image.open(args.image).convert("RGB").resize((512, 512))
    orig_np = np.array(orig, dtype=np.uint8)
    blurred_np = fast_blur.blur(orig_np, radius=RADIUS)
    orig.save("benchmarks/sample_original.png")
    Image.fromarray(blurred_np).save("benchmarks/sample_blurred.png")
    print(f"\n✅ Saved → benchmarks/sample_original.png & sample_blurred.png")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
labels = [f"{s}×{s}" for s in sizes]
x = np.arange(len(sizes))
w = 0.25

speedups_py  = [p / c for p, c in zip(py_times,  cpp_times)]
speedups_pil = [p / c for p, c in zip(pil_times, cpp_times)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"Box Blur Benchmark — radius={RADIUS}", fontsize=14, fontweight="bold")

ax1.bar(x - w, py_times,  w, label="Pure Python", color="#E57373")
ax1.bar(x,     pil_times, w, label="PIL (C)",     color="#FFB74D")
ax1.bar(x + w, cpp_times, w, label="C++ pybind11",color="#4CAF50")
ax1.set_xticks(x); ax1.set_xticklabels(labels)
ax1.set_ylabel("Time (seconds)"); ax1.set_title("Execution Time (log scale)")
ax1.set_yscale("log"); ax1.legend()

b1 = ax2.bar(x - 0.2, speedups_py,  0.38, label="vs Python", color="#42A5F5")
b2 = ax2.bar(x + 0.2, speedups_pil, 0.38, label="vs PIL",    color="#AB47BC")
ax2.set_xticks(x); ax2.set_xticklabels(labels)
ax2.set_ylabel("Speedup (×)"); ax2.set_title("Speedup Factor")
ax2.axhline(y=1, color="red", linestyle="--", alpha=0.4)
ax2.legend()

for bar, val in zip(b1, speedups_py):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(speedups_py)*0.02,
             f"{val:.0f}×", ha="center", fontsize=9, fontweight="bold", color="#1565C0")
for bar, val in zip(b2, speedups_pil):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(speedups_pil)*0.02,
             f"{val:.2f}×", ha="center", fontsize=9, fontweight="bold", color="#6A1B9A")

plt.tight_layout()
plt.savefig("benchmarks/speedup.png", dpi=150)
print("Chart saved : benchmarks/speedup.png")
plt.show()
