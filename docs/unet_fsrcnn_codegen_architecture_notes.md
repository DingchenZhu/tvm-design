
# NVIDIA UNet/FSRCNN Accelerator Instruction & Codegen Architecture

## 1. Background & Goal

This document captures the key **concepts, design decisions, and tech routes** behind the current instruction set and codegen flow for the UNet + FSRCNN accelerator.  
It is intended to help new developers quickly understand:

- What the hardware expects (micro–instructions & buffers)
- How the existing Python codegen scripts work
- Where behavior is currently *hard‑coded*
- How we can move toward a **general front‑end + template‑based back‑end** (PyTorch/ONNX → Relay → Instructions)

The content is distilled mainly from:

- `指令.xlsx` – instruction formats and field semantics  
- `79469e64-..._计算过程.pdf` – detailed hand‑derived compute / tiling schedules  
- `instruction.py` – Python abstraction of the hardware ISA  
- `sd_codegen.py` – UNet (SD) specific codegen  
- `sd_sr_codegen.py` – SR/FSRCNN specific codegen

---

## 2. Hardware Model & Instruction Set (from 指令.xlsx)

The accelerator is designed around a **streaming CNN pipeline** for UNet + FSRCNN.  
All data movement and computation is driven by a small number of micro‑instruction types:

- **OffchipDataLoaderIns** – DDR → offchip_input_buffer / weight_buffer / quant_buffer
- **DataLoaderIns** – input_buffer(a/b/offchip) → line_buffer (with row/column padding + reshape)
- **WeightLoaderIns** – weight_buffer/DDR → MAC array (with bilinear/bicubic, column padding)
- **OffsetLoaderIns** – DDR → offset registers (for deformable / interpolation)
- **QuantLoaderIns** – quant/zero‑point values → quant_reg / zero_point_reg
- **DataStorerIns** – MAC outputs → on‑chip buffers (pooling, pixelshuffle, quant selection)
- **OffchipDataStorerIns** – on‑chip outputs → DDR

Each instruction is essentially a **micro‑DMA or compute configuration** with fields such as:

- `bas_addr`, `transnum`, `stride` – address + length control
- `src_buffer_idx`, `dest_buffer_idx` – which buffer to read/write
- `is_padding_row`, `is_padding_col` – row/column padding semantics
- `line_buffer_reshape`, `line_buffer_row_shift` – how multi‑row data is packed into line buffers
- `weight_parall_mode`, `weight_bit_mode` – weight loading parallelism and bit width
- `quant_mode`, `quant_config_idx` – how quant parameters are broadcast across output channels
- `acc_mode`, `pooling_out_mode`, `pixelshuffle_out_mode` – how MAC results are combined, pooled, and reordered

**Key point:**  
`指令.xlsx` defines the *ISA* and semantics of all these fields. It does **not** prescribe model‑specific values; it is reusable across models.

---

## 3. Manual Scheduling & Tiling (from 计算过程.pdf)

The PDF describes **how to drive the hardware for specific tensor shapes** using these instructions.

Typical patterns:

- Spatial tiling:
  - Height: process **4 or 6 rows** at a time (e.g., one load moves 4 rows into line_buffer).
  - Width: process tiles of **32 / 64 / 128** columns (e.g., output shapes like `4(h) × 32(w) × 8(oc)`).
  - Large images (e.g., 256×144) are split into sub‑images (e.g., two 128×144 halves) and processed independently.

- Channel tiling:
  - Input channels (`ic`) grouped as **2ic / 4ic / 8ic** per loop; multiple loops accumulate into the same `acc_reg`.
  - Output channels (`oc`) grouped (e.g., **8 / 16 / 32 oc**), sometimes split across upper/lower halves of the MAC array and combined using `acc_mode`.

- Border handling:
  - **Row padding** uses `is_padding_row` with different codes for:
    - first rows, last rows, repeated edge rows, all‑zero rows, etc.
  - **Column padding** uses `is_padding_col` (0–6) to control how many valid values vs zeros per row.

- Layout transforms:
  - Different **line_buffer load modes** (0–3/4) decide how many rows per cycle and how values are interleaved.
  - **Pooling** and **pixelshuffle** patterns are specified via `pooling_out_mode`, `pixelshuffle_out_mode`, with detailed descriptions on how 128 values are grouped and interleaved.

This PDF is effectively a catalog of **hand‑derived tiling / scheduling templates** for specific shapes:

- “大小为 (64,36)”
- “大小为 (32,18)”
- “大小为 (256,144) + concat”
- Cases like `(8,16,3)`, `(4,8,3)`, `(32,4,3)` etc.

Each case shows:

1. How inputs are loaded & rearranged
2. How weights are loaded (`ky <- ic <- kx <- oc`)
3. How many cycles to get one row / several rows of output
4. How to store to on‑chip buffers with correct strides and grouping
5. Example `dataloader/weightloader/datastorer` parameter settings

---

## 4. Python ISA Wrapper (instruction.py)

`instruction.py` defines a **thin, generic Python API** for emitting hardware instructions.

- A global `Inst` class holds:
  - `code_list`: list of all instructions generated in a run
  - `current_code_num`: monotonically increasing instruction ID

- For each hardware opcode there is a **Python wrapper class**:
  - `OffchipDataLoader`
  - `DataLoader`
  - `WeightLoader`
  - `OffsetLoader`
  - `QuantLoader`
  - `DataStorer`
  - `OffchipDataStorer`

- Each wrapper exposes a `dispatch(...)` method:
  - Takes keyword args corresponding to the fields in `指令.xlsx`
  - Builds a Python dict:
    ```python
    code = {
        "code_num": [Inst.current_code_num],
        "op_code": "DataLoader",
        "layer_idx": layer_idx,
        "bas_addr": bas_addr,
        "transnum": transnum,
        ...
    }
    Inst.code_list.append(code)
    Inst.current_code_num += 1
    ```

**Important:**

- `instruction.py` is **ISA‑level only**:
  - It does **not** know about UNet layers, tensor shapes, or tiling.
  - It is *not* where anything model‑specific is hard‑coded.
- This module is reusable for any model and any compiler backend that targets this hardware.

---

## 5. UNet Codegen (sd_codegen.py)

`sd_codegen.py` is a **model‑specific backend codegen** for the UNet part of the pipeline.

### 5.1 Layer configuration is hard‑coded

At the top there is a `layer_config` list like:

```python
layer_config = [
    {"layer_index": 0, "c_in": 1, "c_out": 4, "kernel_size": 3, "groups": 1},
    ...
    {"layer_index": 18, "c_in": 4, "c_out": 1, "kernel_size": 3, "groups": 1}
]
```

This encodes the entire UNet’s channel structure **by hand**.  
There is no automatic import from PyTorch/ONNX/Relay.

### 5.2 Resource managers (generic but used in a fixed way)

The script defines several “manager” classes:

- `DataLoaderManager`  
  Tracks:
  - Which input buffer (a/b/offchip) is active
  - Current `line_buffer_idx`
  - Current base addresses for loading

- `WeightLoaderManager`  
  Tracks:
  - Which line buffer and acc_reg bank are mapped
  - Base addresses for different `weight_parall_mode` options

- `QuantLoaderManager`  
  Tracks:
  - Quant buffer base address
  - Current quant config index

- `DataStorerManager`  
  Tracks:
  - Destination buffer (a/b/output/unet output)
  - Current result base address
  - Quant config index and simple buffer modeling

These managers are fairly generic and reusable: they encode **resource state**, not a specific model.

### 5.3 Main code path: per‑layer hand‑written schedule

In `if __name__ == "__main__":` the script:

1. Issues initial `OffchipDataLoader` instructions to bring:
   - All UNet quant parameters
   - All UNet weights
   - Input image(s)
   into on‑chip/offchip buffers.

2. For each UNet layer, it then **hand‑codes**:

   - How many rows/tiles to process:
     ```python
     cal_total_num = 144 // 2    # each time compute 2 rows
     load_total_num = 144 // 2
     padding_num = 1
     ```
   - How `is_padding_row` is set for head/middle/tail loads:
     ```python
     if load_idx < padding_num: is_padding_row = 1
     elif load_idx > load_total_num - 1 - padding_num: is_padding_row = 5
     else: is_padding_row = 0
     ```
   - How `bas_addr` increments for each load:
     ```python
     if load_idx < padding_num:
         dataloadermanager.bas_addr_cur += 2
     else:
         dataloadermanager.bas_addr_cur += 4
     ```
   - Hard‑coded constants like image width, strides, tile sizes and magic address formulas:
     ```python
     datastorermanager.base_addrs_res_cur = 0
     stride = 144
     # many addresses like 144*4*2 + 72*8, etc.
     ```

   - For each tile, it calls:
     - `DataLoader.dispatch(...)`
     - `WeightLoader.dispatch(...)`
     - `DataStorer.dispatch(...)`
     with fixed field values derived from the manual schedule.

3. At the end, it post‑processes `Inst.code_list` to:
   - Add data dependencies between instructions
   - Assign virtual registers and life ranges (`src1/src2/src3/dest`)

### 5.4 Where things are “written in stone”

**Hard‑coded (per‑model, per‑resolution) in `sd_codegen.py`:**

- UNet layer topology (`layer_config`)
- Expected input size(s): e.g., 256×144, 128×144 halves, etc.
- Tiling strategy:
  - How to split into left/right halves
  - How many rows per tile (4/6 rows)
  - How many tiles per layer
- Exact parameters for:
  - `transnum`, `is_padding_row`, `line_buffer_reshape`, `read_mode`
  - `weight_parall_mode`, `is_padding_col`, `line_buffer_row_shift`
  - `store_mode`, `acc_mode`, `quant_mode`, `pixelshuffle_out_mode`
- All base address formulas (`bas_addr`, `base_addrs_res_cur`, etc.)

In other words, `sd_codegen.py` is a **hand‑crafted scheduler + codegen for one specific UNet configuration and input resolution**.

---

## 6. SR / FSRCNN Codegen (sd_sr_codegen.py)

`sd_sr_codegen.py` plays the same role as `sd_codegen.py`, but for the **super‑resolution / FSRCNN** part and some UNet offset/bicubic paths.

Key points:

- It defines the same family of managers plus an `OffsetLoaderManager`.
- It initializes base addresses for SR weights and quant parameters, *taking into account* space already used by the UNet:
  ```python
  WeightLoaderManager.bas_addr_cur = [1737, 792, 1152]
  QuantLoaderManager.bas_addr_cur = 665
  ```
- The main function `sd_inst(is_first=False, load_next=True)`:
  - Optionally loads all UNet quant/weights/images when `is_first=True`.
  - Then executes a **sequence of layer‑specific schedules** for SR, mirroring the cases in the PDF:
    - `(32,32,...)`, `(128,72,...)+concat`, `(256,144)+concat`, etc.
  - For each case:
    - It loops over tiles (H/W/IC/OC groups)
    - Sets `is_padding_row`, `line_buffer_reshape`, `transnum`, `weight_parall_mode`, `is_bilinear`, `pixelshuffle_mode_out`, etc.
    - Updates `bas_addr` and other manager states manually.

**Again, the tiling and addresses are completely hard‑coded**, but the emitting of instructions still goes through the generic wrappers in `instruction.py`.

---

## 7. Where Each Layer of Logic Lives

Putting it together in terms of a generic pipeline:

```text
PyTorch / ONNX
    ↓
Relay IR
    ↓           (not implemented in these files)
Layer Description List
(ops, shapes, strides, groups, etc.)
    ↓
Schedule & Tiling
(model-specific; currently hard-coded in sd_codegen*.py)
    ↓
Instruction Templates
(implemented manually using the patterns from 计算过程.pdf)
    ↓
instruction.py
(generic ISA wrappers → Inst.code_list)
    ↓
Binary program for accelerator
```

- **ISA Definition**: `指令.xlsx` + `instruction.py` → *generic, reusable*
- **Manual Tiling/Scheduling**: `79469e64_计算过程.pdf` → *reference templates*
- **Model‑specific Backends**: `sd_codegen.py`, `sd_sr_codegen.py` → *where everything is currently baked in*

---

## 8. What a “General Front‑End” Can & Should Do

Your original question:  
> Can we design a general front‑end (PyTorch/ONNX → Relay → Instructions) without hard‑coding all parameters, or must we fix them during front‑end integration?

Based on the current codebase:

### 8.1 What should remain *fixed* (by hardware design)

These are “ISA facts” and should **not change per model**:

- Instruction opcodes & field sets (what each instruction type means).
- Enumerations and semantics:
  - `is_padding_row` / `is_padding_col` codes
  - `line_buffer_reshape` values
  - `weight_parall_mode`, `quant_mode`, `acc_mode`, `pooling_out_mode`, `pixelshuffle_out_mode`, etc.
- Supported line_buffer / weight deployment / store modes (essentially the primitive tiling patterns the hardware can execute).

These are correctly centralized in `指令.xlsx` and `instruction.py`.

### 8.2 What is currently hard‑coded but *should* be generated

All of the following ideally come from a **compiler pass**, *not* from hand‑written Python:

- Per‑layer topology (`c_in`, `c_out`, `kernel_size`, `groups`, activation shapes)
  - Should be derived automatically from Relay IR.
- Tiling decisions per layer:
  - How to decompose H/W into tiles (e.g., 4h×32w vs 4h×64w)
  - How to decompose IC/OC into groups
  - When to split wide images into 128‑wide halves, etc.
- Exact instruction fields:
  - `bas_addr`, `transnum`, `stride`
  - `is_padding_row`, `is_padding_col`
  - `line_buffer_reshape`, `read_mode`
  - `weight_parall_mode`, `acc_mode`, `quant_mode`, `pixelshuffle_out_mode`, etc.

These are what `sd_codegen.py` and `sd_sr_codegen.py` currently fix “by hand”.

### 8.3 What the *front‑end* can concretely do

A more scalable design:

1. **Relay → LayerDesc pass (front‑end side)**
   - For each op in Relay, produce a neutral description:
     ```python
     {
       "op": "conv2d",
       "layer_idx": i,
       "H_in": H_in,
       "W_in": W_in,
       "Cin": Cin,
       "Cout": Cout,
       "kernel": 3,
       "stride": (1,1),
       "padding": "same" / explicit,
       "groups": groups,
       "is_deformable": bool,
       "need_pixelshuffle": bool,
       ...
     }
     ```
   - This is **model‑generic** and does *not* touch any ISA details.
   - It replaces the hard‑coded `layer_config` in `sd_codegen.py`.

2. **Backend “template library” (replacing scattered handwritten code)**
   - Encapsulate the patterns from `计算过程.pdf` as reusable functions, e.g.:
     ```python
     def emit_conv3x3_tile(desc, tiling_cfg, managers):
         # internally emit DataLoader/WeightLoader/DataStorer using instruction.py
     ```
   - `tiling_cfg` may include:
     - Tile size (H_tile, W_tile)
     - IC/OC grouping sizes
     - Which line_buffer / weight / store mode to use
   - Most of the “magic numbers” in `sd_codegen*.py` become formulas in these template functions.

3. **Automatic scheduler (between Relay and templates)**
   - Given a `LayerDesc` and hardware constraints:
     - Choose a tiling template (e.g., 4×32, 4×64, 4×128, deformable variant…)
     - Determine number of tiles in H/W/IC/OC
     - Calculate base addresses and strides using generic formulas
   - This logic **replaces** the long per‑layer loops in `sd_codegen.py` and `sd_sr_codegen.py`.

4. **ISA emission (unchanged)**
   - Continue to use `instruction.py`’s `dispatch()` API to append instructions to `Inst.code_list`.

---

## 9. Required Knowledge & Skills

Developers working on this stack should be familiar with:

- **Hardware/architecture side**
  - CNN accelerator micro‑architecture: line buffers, MAC arrays, double buffering
  - Memory hierarchy: offchip DDR vs on‑chip input/weight/quant/output buffers
  - Tiling strategies and trade‑offs (H/W/IC/OC)

- **Compiler / IR side**
  - Relay IR (TVM): graph representation, shape inference, operator attributes
  - Basic scheduling concepts: loop tiling, fusion, unrolling, data reuse
  - Instruction selection and register allocation (for the minimal “virtual register” logic)

- **Codebase specifics**
  - Semantics of each micro‑instruction field (`指令.xlsx`)
  - Patterns documented in `计算过程.pdf`
  - Python structure of `instruction.py` and the manager classes in `sd_codegen*.py`

---

## 10. Recommended Tech Route (Summary)

1. **Keep `instruction.py` as the stable ISA layer.**
2. **Extract a formal “hardware primitive template library”** from the patterns in `计算过程.pdf` and the current `sd_codegen*.py`.
3. **Introduce a Relay → LayerDesc pass (front‑end)** that:
   - Extracts layer shapes and attributes from arbitrary models (PyTorch/ONNX → Relay).
4. **Implement a scheduler that:**
   - Chooses tiling templates per layer
   - Computes all instruction parameters programmatically
   - Uses the template library + managers to emit instructions via `instruction.py`.
5. **Gradually delete hand‑coded per‑layer logic** in `sd_codegen.py` and `sd_sr_codegen.py`, replacing it with template‑driven calls.

This way:

- The **front‑end remains general** (no ISA fields or per‑model magic numbers).
- The **back‑end encodes hardware‑specific knowledge** (tiling templates, address formulas) in a structured, reusable way.
- Adding a new model or changing resolution becomes a matter of:
  - Supporting it in PyTorch/ONNX → Relay
  - Ensuring the scheduler can cover its shapes with existing templates (or adding a new template),  
  rather than re‑writing a large hand‑tuned codegen script.
