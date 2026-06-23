import sys, time, os, argparse

# Setup paths to ensure we can import the local build modules
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "build"))

import numpy as np
import matplotlib.pyplot as plt
import fast_blur
from python_impl.blur import box_blur
from PIL import Image, ImageFilter


import fast_blur
from python_impl.blur import box_blur

def get_image(size, image_path=None):
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB").resize((size, size))
        return np.array(img, dtype=np.uint8)
    # Fallback to random noise if no valid image is provided
    return np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark fast_blur against pure Python and PIL")
    parser.add_argument("--image", type=str, default=None, help="Path to a real image to use instead of synthetic data")
    parser.add_argument("--radius", type=float, default=3.0, help="Blur radius")
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512, 1024, 2048], help="List of square image sizes to benchmark")
    args = parser.parse_args()

    radius = args.radius
    sizes = args.sizes

    print(f"\nRadius: {radius} | Source: {args.image or 'synthetic'}\n")
    print(f"{'Size':<10} {'Python':>10} {'PIL':>10} {'C++':>10} {'vs Python':>12} {'vs PIL':>10}")
    print("-" * 68)

    py_times, pil_times, cpp_times = [], [], []

    for s in sizes:
        img_np = get_image(s, args.image)
        img_pil = Image.fromarray(img_np)
        flat_bytes = img_np.flatten().tobytes()

        # Pure Python is generally slow, so we only measure it once
        start = time.perf_counter()
        box_blur(flat_bytes, s, s, 3, int(radius))
        py_t = time.perf_counter() - start

        # Average PIL over 3 runs
        pil_runs = []
        for _ in range(3):
            start = time.perf_counter()
            img_pil.filter(ImageFilter.BoxBlur(radius))
            pil_runs.append(time.perf_counter() - start)
        pil_t = sum(pil_runs) / 3

        # Average C++ pybind11 over 3 runs
        cpp_runs = []
        for _ in range(3):
            start = time.perf_counter()
            fast_blur.blur(img_np, radius=radius)
            cpp_runs.append(time.perf_counter() - start)
        cpp_t = sum(cpp_runs) / 3

        py_times.append(py_t)
        pil_times.append(pil_t)
        cpp_times.append(cpp_t)

        vs_py = py_t / cpp_t
        vs_pil = pil_t / cpp_t
        print(f"{s}x{s:<7} {py_t:>9.3f}s {pil_t:>9.4f}s {cpp_t:>9.4f}s {vs_py:>10.1f}x {vs_pil:>9.2f}x")

    # Ensure output directory exists before saving
    os.makedirs("benchmarks", exist_ok=True)

    # Save an example of the blurred output if a real image was used
    if args.image and os.path.exists(args.image):
        orig = Image.open(args.image).convert("RGB").resize((512, 512))
        orig_np = np.array(orig, dtype=np.uint8)
        blurred_np = fast_blur.blur(orig_np, radius=radius)
        
        orig.save("benchmarks/sample_original.png")
        Image.fromarray(blurred_np).save("benchmarks/sample_blurred.png")
        print(f"\n Saved image samples to benchmarks/")

    # Plotting the benchmarking results
    labels = [f"{s}x{s}" for s in sizes]
    x = np.arange(len(sizes))
    w = 0.25

    speedups_py = [p / c for p, c in zip(py_times, cpp_times)]
    speedups_pil = [p / c for p, c in zip(pil_times, cpp_times)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Box Blur Benchmark — radius={radius}", fontsize=14, fontweight="bold")

    ax1.bar(x - w, py_times, w, label="Pure Python", color="#E57373")
    ax1.bar(x, pil_times, w, label="PIL (C)", color="#FFB74D")
    ax1.bar(x + w, cpp_times, w, label="C++ pybind11", color="#4CAF50")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("Execution Time (log scale)")
    ax1.set_yscale("log")
    ax1.legend()

    b1 = ax2.bar(x - 0.2, speedups_py, 0.38, label="vs Python", color="#42A5F5")
    b2 = ax2.bar(x + 0.2, speedups_pil, 0.38, label="vs PIL", color="#AB47BC")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Speedup (x)")
    ax2.set_title("Speedup Factor")
    ax2.axhline(y=1, color="red", linestyle="--", alpha=0.4)
    ax2.legend()

    # Add numeric labels on top of the speedup bars
    for bar, val in zip(b1, speedups_py):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(speedups_py) * 0.02,
                 f"{val:.0f}x", ha="center", fontsize=9, fontweight="bold", color="#1565C0")
        
    for bar, val in zip(b2, speedups_pil):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(speedups_pil) * 0.02,
                 f"{val:.2f}x", ha="center", fontsize=9, fontweight="bold", color="#6A1B9A")

    plt.tight_layout()
    plt.savefig("benchmarks/speedup.png", dpi=150)
    print("Chart saved to benchmarks/speedup.png")
    plt.show()