#!/usr/bin/env python3
import subprocess, os, sys, time, shlex
import numpy as np
import matplotlib.pyplot as plt
import csv

runs = 50
kernels = {
    'gemv': 'gemv.c',
    'gemm': 'gemm.c',
    'conv2d': 'conv2d.c',
    'activations_relu': 'activations.c',  # will set env OP=relu
    'activations_sigmoid': 'activations.c' # OP=sigmoid
}

builds = {
    'scalar': ['-O2', '-o'],
    'auto': ['-O3', '-march=native', '-o'],
    'avx2': ['-O3', '-mavx2', '-mfma', '-march=native', '-DUSE_AVX2', '-o']
}

def compile_exe(kernel_src, outname, flags, extra=[]):
    cmd = ['gcc', kernel_src] + flags + extra
    print("Compiling:", " ".join(cmd))
    subprocess.check_call(cmd)

def run_exe(path, env=None):
    proc = subprocess.run([path], capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print("Error running", path)
        print(proc.stderr)
        raise RuntimeError("exec failed")
    return float(proc.stdout.strip())

def ensure_builds():
    for kname, src in kernels.items():
        for bname, flags_tpl in builds.items():
            exe_name = f"{kname}_{bname}"
            # decide extra flags and output name position
            flags = []
            if bname == 'scalar':
                flags = ['-O2', '-o', exe_name]
            elif bname == 'auto':
                flags = ['-O3', '-march=native', '-o', exe_name]
            elif bname == 'avx2':
                flags = ['-O3','-mavx2','-mfma','-march=native','-DUSE_AVX2','-o', exe_name]
            # some kernels need math lib
            extra = []
            if src.endswith('.c'):
                extra = []
            compile_exe(src, exe_name, flags + ["-lm"], extra)

def bench_all():
    results = {}
    for kname in kernels.keys():
        results[kname] = {}
        for bname in builds.keys():
            exe = f"./{kname}_{bname}"
            print(f"Running {exe} {runs} times...")
            times = []
            for r in range(runs):
                env = os.environ.copy()
                # activations choose op via env var
                if kname.endswith('relu'):
                    env['OP']='relu'
                if kname.endswith('sigmoid'):
                    env['OP']='sigmoid'
                t = run_exe(exe, env=env)
                times.append(t)
            arr = np.array(times)
            results[kname][bname] = arr
            # save csv
            csvfn = f"{kname}_{bname}_times.csv"
            with open(csvfn,'w',newline='') as f:
                w = csv.writer(f)
                w.writerow(['run','time_s'])
                for i,val in enumerate(arr.tolist(),1):
                    w.writerow([i,val])
            print(f"Saved {csvfn}")
    return results

def plot_results(results):
    for kname, data in results.items():
        plt.figure(figsize=(10,6))
        for bname, arr in data.items():
            plt.plot(arr, label=f"{bname} mean={arr.mean():.6f}s")
        plt.title(f"{kname} timings ({runs} runs)")
        plt.xlabel("Run")
        plt.ylabel("Time (s)")
        plt.legend()
        plt.grid(True)
        png = f"{kname}_plot.png"
        plt.savefig(png)
        print("Saved plot", png)
        plt.show()

def print_stats(results):
    for kname, data in results.items():
        print("\n=== Kernel:", kname)
        base_mean = data['scalar'].mean()
        for bname, arr in data.items():
            print(f" {bname:6s} mean={arr.mean():.6f} med={np.median(arr):.6f} std={arr.std():.6f}")
        print(" Speedups over scalar (mean):")
        for bname in ['auto','avx2']:
            print(f"  {bname:6s} {base_mean/data[bname].mean():.2f}x")

if __name__=='__main__':
    ensure_builds()
    results = bench_all()
    print_stats(results)
    plot_results(results)
