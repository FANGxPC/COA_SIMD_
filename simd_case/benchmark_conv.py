import subprocess
import numpy as np
import matplotlib.pyplot as plt

def run_executable(exe, runs=50):
    times = []
    for _ in range(runs):
        result = subprocess.run([f"./{exe}"], capture_output=True, text=True)
        t = float(result.stdout.strip())
        times.append(t)
    return np.array(times)

executables = ["conv_scalar", "conv_auto", "conv_avx2"]
results = {}

for exe in executables:
    print(f"Running {exe} ...")
    results[exe] = run_executable(exe)

plt.figure(figsize=(10, 6))
for exe, times in results.items():
    plt.plot(times, label=f"{exe} (mean={np.mean(times):.4f}s)")

plt.title("3×3 Convolution Benchmark (50 runs)")
plt.xlabel("Run number")
plt.ylabel("Execution time (s)")
plt.legend()
plt.grid(True)
plt.show()

for exe, times in results.items():
    print(f"\n{exe}:")
    print(f"  Mean   = {np.mean(times):.6f} s")
    print(f"  Median = {np.median(times):.6f} s")
    print(f"  Stddev = {np.std(times):.6f} s")

print("\nSpeedup (relative to scalar):")
base = np.mean(results["conv_scalar"])
for exe in executables[1:]:
    speedup = base / np.mean(results[exe])
    print(f"  {exe}: {speedup:.2f}×")
