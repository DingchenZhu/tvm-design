# How to Know If Your Instructions Match Golden on Hardware (Functionality & Performance)

**Short answer:** You cannot know from this repo alone. You need to run **both** your instructions and the golden on the **same platform** (VPU simulator or real hardware) and compare results and performance.

---

## 1. Functionality (same outputs for same inputs)

To prove that your instruction stream produces the **same function** as the golden on the hardware:

1. **Get a reference output** (ground truth):
   - Run the **reference model** (ONNX or PyTorch USRNet) on a **fixed test input** (e.g. one 256×256 image).
   - Save the output tensor(s) to a file (e.g. NumPy `.npy` or `.npz`).

2. **Run your instructions on the platform**:
   - Load your `usr_net_inst.bin` (and parameters) into the **VPU simulator or real hardware**.
   - Feed the **same test input**.
   - Capture the output tensor(s) produced by the hardware/simulator and save to a file.

3. **Run the golden instructions on the same platform** (optional but recommended):
   - Load the golden instruction stream (and its parameters) into the **same** VPU simulator or hardware.
   - Feed the **same test input**.
   - Capture the golden output and save to a file.

4. **Compare outputs**:
   - Use the script `compare_reference_with_vpu_output.py` (in this folder) to compare:
     - Reference (ONNX/PyTorch) vs your VPU output → confirms your compiler is correct.
     - Reference vs golden VPU output → confirms golden is correct.
     - Your VPU output vs golden VPU output → confirms your instructions match golden on the platform.

If max absolute difference and relative error are within tolerance (e.g. quantization error), **functionality is equivalent**.

**This repo does not include a VPU simulator.** You need either:
- A VIS VPU simulator (RTL or behavioral) that can load `.bin` and run with test data, or  
- Real VIS VPU hardware with a way to load instructions and dump output buffers.

---

## 2. Performance (same or acceptable cycles / throughput)

To know if your instructions achieve the **same (or acceptable) performance** as the golden on the hardware:

1. **On simulator or hardware**:
   - Run **your** instruction stream and record **cycle count** (and optionally memory bandwidth).
   - Run the **golden** instruction stream with the **same input size** and record cycle count.
   - Compare: same or lower cycles for your stream → performance is at least as good.

2. **Without simulator/hardware (estimation only)**:
   - Use the script `estimate_cycles.py` (in this folder) to get an **estimated** cycle count for both your instructions and the golden, using a simple per-op cycle model (e.g. from `VERIFICATION_REPORT.md`).
   - This gives a **relative comparison** (e.g. “your stream has fewer estimated cycles than golden”) but **not** real platform performance. Real performance depends on:
     - Pipeline and parallelism
     - Memory latency and bandwidth
     - How well the scheduler uses dependency info (golden has full dependency; yours may not)

So:
- **Estimated cycles** (script): use to compare “shape” of workload (your vs golden).
- **Real cycles** (simulator/hardware): only way to know actual performance on the platform.

---

## 3. Summary

| Question | How to answer |
|----------|----------------|
| Do my instructions compute the same function as the golden on hardware? | Run both on VPU simulator/hardware with same input; compare output tensors (e.g. with `compare_reference_with_vpu_output.py`). |
| Do my instructions get the same performance as the golden on hardware? | Run both on VPU simulator/hardware; compare cycle count (and throughput). Without platform: use `estimate_cycles.py` for estimated comparison only. |
| Can I know this without a simulator or hardware? | **No.** This repo only does structural/instruction-level checks and cycle **estimation**. |

---

## 4. Scripts in this folder

- **`compare_with_golden.py`** – Structural comparison of your vs golden instruction streams (logical ops, counts). Does **not** run anything.
- **`compare_reference_with_vpu_output.py`** – Compares reference (NumPy) output with VPU output file(s). Use **after** you have run the model on a simulator/hardware and saved outputs.
- **`estimate_cycles.py`** – Estimates total cycles for an instruction stream (and optionally compares yours vs golden). Use for a **rough performance comparison** when you don’t have simulator/hardware.
