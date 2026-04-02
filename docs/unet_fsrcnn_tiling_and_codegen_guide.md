# UNet + FSRCNN Accelerator Tiling & Codegen Design

> 目标：**只读这一份文档**，就能理解这套加速器的：
>
> - 指令集（Instructions）与硬件数据通路
> - 每一类算子（尤其是 UNet / FSRCNN）如何做 tiling
> - PDF《计算过程》和 Excel《指令.xlsx》里那些 case 的含义
> - 三个代码文件 `instruction.py / sd_codegen.py / sd_sr_codegen.py` 分别在做什么、在哪一层“写死”了什么
> - 如果要做通用前端（PyTorch/ONNX → Relay → Instructions），应该怎么设计接口和后端模板

---

## 1. 整体数据流 & 编译流程

### 1.1 硬件数据流（宏观）

核心模块和 buffer：

- **片外 DDR**
- **offchip_input_buffer / weight_buffer / quant_buffer**：片上 SRAM，用于缓存从 DDR 拉上来的输入 / 权重 / 量化参数。
- **input buffer a / b**：片上 feature map buffer，用来在层与层之间传递数据。
- **line_buffer（双路）**：给 MAC 阵列准备窗口数据（行缓冲 + 重排 + padding）。
- **MAC 阵列 + acc_reg（双路）**：执行卷积 / 插值 / pooling 的核心。
- **fsrcnn_output_buffer / unet_output_buffer / offset_reg**：各自 pipeline 的最终输出或中间寄存器。

数据通路分成几个阶段：

1. **OffchipDataLoader**：DDR → offchip_input_buffer / weight_buffer / quant_buffer。
2. **DataLoader**：offchip_input_buffer 或 input a/b → line_buffer（带重排 + 行/列 padding）。
3. **WeightLoader / OffsetLoader / QuantLoader**：加载权重、offset、量化参数，准备好 MAC。
4. **MAC 计算**：在 acc_reg 中得到中间结果。
5. **DataStorer**：对 acc_reg 输出做 pooling / pixelshuffle / 量化，写回 input a/b 或输出 buffer。
6. **OffchipDataStorer**：最终结果从输出 buffer 回写 DDR。

### 1.2 编译流程（理想化）

目标流程（高层）：

```text
PyTorch / ONNX
      ↓
  Relay IR
      ↓             (前端)
Layer 描述 (shape, op, stride, group, ... )
      ↓             (调度 / 后端)
Tiling 决策 + 模板选择 + 地址计算
      ↓
微指令流 (OffchipDataLoader / DataLoader / WeightLoader / ...)
```

当前仓库里的三个文件对应：

- `instruction.py`：**指令 ISA 定义 + 容器**（纯后端工具库，通用）。
- `sd_codegen.py`：**UNet（sd）网络的手写 codegen + 调度**（把特定模型 + 分辨率 + tiling 全部写死）。
- `sd_sr_codegen.py`：**SR / FSRCNN + offset pipeline 的手写 codegen + 调度**。

---

## 2. 指令集概览（源于《指令.xlsx》与 `instruction.py`）

Excel 里的 `OffchipDataLoaderIns / DataLoaderIns / WeightLoaderIns / OffsetLoaderIns / QuantLoaderIns / DataStorerIns / OffchipDataStorerIns`，在 `instruction.py` 里对应 7 个类：

```python
class OffchipDataLoader:
    dispatch(transnum, load_model, src_buffer_idx, bas_addr)

class DataLoader:
    dispatch(layer_idx, line_buffer_reshape, is_padding_row, read_mode,
             transnum, line_buffer_idx, src_buffer_idx, bas_addr)

class WeightLoader:
    dispatch(acc_reg_comp_idx, kernal_size, line_buffer_row_shift, line_buffer_idx,
             is_padding_col, weight_parall_mode, is_new, transnum, bas_addr,
             is_bilinear_bicubic, offset_reg_idx)

class OffsetLoader:
    dispatch(offset_reg_idx, bas_addr)

class QuantLoader:
    dispatch(quant_reg_load_idx, quant_mode, layer_idx, transnum, bas_addr)

class DataStorer:
    dispatch(quant_config_idx, pixelshuffle_out_mode, is_pixelshuffle,
             pooling_out_mode, pooling_out_new, is_pooling,
             reg_out_idx, acc_mode, transfer_num, store_mode, stride,
             base_addr_pooling, base_addrs_res,
             is_bicubic_add, is_first_or_last_row, is_mask, is_new, dest_buffer_idx)

class OffchipDataStorer:
    dispatch(src_buffer, transnum, base_addr)
```

### 2.1 字段含义（与 Excel 中的定义一一对应）

这里只给最关键的语义，详细枚举值可以回看《指令.xlsx》。

#### OffchipDataLoader

- `transnum`：一次指令要从 DDR 读多少“块”。每块大小由硬件固定 + `load_model` 决定（如 64cycle/32cycle 等）。
- `load_model`：**加载模式**，Excel 里定义：
  - 0：UNet 输入，每次 64cycle 直接写入 offchip_input_buffer。
  - 1：FSRCNN 输入，32 个数拆成多个地址写入（重排）。
  - 还有模式用于 weight_buffer / quant_buffer。
- `src_buffer_idx`：目标 buffer（0: offchip_input_buffer, 1: weight_buffer, 2: quant_buffer…）。
- `bas_addr`：片外（或目标 buffer 中）起始地址。

#### DataLoader

- `layer_idx`：标记当前层，方便硬件做 debug / profiling。
- `line_buffer_reshape`：**line_buffer 加载模式**，Excel 里几个典型：
  - 0：连续 load 多行，直接填 line_buffer。
  - 1：前 2cycle 填四行前 64，后 2cycle 填四行后 64。
  - 2/3：每个 cycle 填四行，每行 8 或 32 个，用于特定稀疏/分组布局。
- `is_padding_row`：行方向 padding 策略，支持：
  - 0 不补零；1/2/3/4/5/6/7 不同 “首行全 0 / 尾行全 0 / 后半为 0 / 边缘复制”等模式。
- `read_mode`：从 input buffer 读出的方式（直接输出 / 两个 cycle 交插 / deformable 特别模式）。
- `transnum`：本次要将多少“行块”写到 line_buffer。
- `line_buffer_idx`：选择哪一路 double line buffer（0/1）。
- `src_buffer_idx`：从哪个 input buffer 读（'a' / 'b' / 'offchip_input_buffer' 等）。
- `bas_addr`：在 src buffer 里的起始地址。

#### WeightLoader

- `acc_reg_comp_idx`：选择哪一路 double acc_reg。
- `kernal_size`：0: k=3, 1: k=1, 2: bicubic (k=4)。
- `line_buffer_row_shift`：行之间的移位方式，配合 k=3/k=1 做行复用。
- `line_buffer_idx`：使用哪一路 line_buffer。
- `is_padding_col`：列补零方式（不补 / 补 128/64/32/16/8 / 32 补两个零等）。
- `weight_parall_mode`：一次加载多少个权重：
  - 0: 8 个；1: 16 个；2: 32 个。
- `is_new`：0: 覆盖 acc_reg；1: 在 acc_reg 基础上累加（用于累加不同 ic group）。
- `transnum`：要加载多少拍（每拍 8/16/32 个）。
- `bas_addr`：权重在 weight_buffer 中的起始地址。
- `is_bilinear_bicubic`：0: 不插值；1: 双线性；2: bicubic。
- `offset_reg_idx`：可变形卷积 / 插值使用哪组 offset_reg。

#### QuantLoader

- `quant_reg_load_idx`：写哪一套 quant_reg / zero_point_reg（支持 double quant reg）。
- `quant_mode`：**1cycle 与输出通道 oc 的对应节奏**，Excel 给出多种模式：
  - 例如 `0: 1cycle 1oc, 2cycle 换 oc`，`1: 1cycle 1oc, 1cycle 换 oc` 等。
- `layer_idx`：当前层。
- `transnum`：要加载多少个 quant 参数。
- `bas_addr`：quant_buffer 中的起始地址。

#### DataStorer

- `quant_config_idx`：选择哪一套 quant 配置（double quant reg）。
- `pixelshuffle_out_mode` + `is_pixelshuffle`：是否做 pixelshuffle，以及 128 个值分成几组、如何插入。
- `pooling_out_mode` + `pooling_out_new` + `is_pooling`：各种 4×4 / 8×8 pooling 模式及是否两次阵列输出相加。
- `reg_out_idx`：选择哪一组 acc_reg 输出。
- `acc_mode`：acc_reg 取数顺序，对应 1/2、1/4、1/8 平均或 bicubic 特殊模式。
- `transfer_num`：本次要写多少“块”结果（块大小由 store_mode 决定）。
- `store_mode`：一次写多少 cycle，例如 4/8/32/64/128。
- `stride`：地址 stride（支持 “2cycle 连续，2cycle 加 stride”、“4cycle 连续，4cycle 加 stride”等）。
- `base_addr_pooling`：pooling 结果的起始地址。
- `base_addrs_res`：最终结果写回 input a/b / output buffer 的起始地址。
- `is_bicubic_add` / `is_first_or_last_row` / `is_mask` / `is_new` / `dest_buffer_idx`：分别控制
  - bicubic 中间行/首尾行的处理；
  - mask 层（量化后只剩 0/1）；
  - 与原值累加与否；
  - 目标 buffer（a/b/fsrcnn/unet/output/offset）。

#### OffchipDataStorer

- `src_buffer`：0: unet output reg；1: fsrcnn output buffer。
- `transnum`：写回多少块。
- `base_addr`：片外地址。

> 小结：**Excel 定义了字段和枚举；`instruction.py` 把这些字段变成统一的 Python API，用 `dispatch()` 来生成一条指令字典并放进 `Inst.code_list`。这里没有“模型信息”和“tiling 决策”，只是 ISA。**

---

## 3. Tiling 总体策略（从 PDF《计算过程》抽象出来）

### 3.1 约束与基本单位

从 PDF 可见的约束（以 UNet 为例）：

- **行方向 H 的基本单位**：4 行（普通卷积）或 6 行（带 bilinear / deformable 时，6 行里取 4 行做插值）。
- **列方向 W 的基本单 tile 宽度**：32 / 64 / 128。
- **输入通道 Cin**：以 2ic / 4ic / 8ic 为 group。
  - 文档中常见：`一个 transnum 中包含 2ic`、`循环 2 次完成 8ic 的切换` 等。
- **输出通道 Cout**：以 8 / 16 / 32 oc 为 group。
  - 比如“前 4 个 oc 和后 4 个 oc 分别计算相同 oc 的不同 ic，最后累加”、“上下 32×8 的阵列分别计算 4 组 oc”。
- **line_buffer**：双路，使用 `line_buffer_reshape` 决定“每 cycle 填几行，每行填多少个”。
- **acc_reg**：双路，使用 `acc_mode` 决定怎样从多个 acc_reg 组合结果（平均 / 特殊加权等）。

### 3.2 空间 tiling：H 与 W

**H 方向：**

- 每次 DataLoader 负责填充 4 行或 6 行 line_buffer。
- 整幅 H 被拆成多个 block，例如 144 高时，经常是：
  - “一次算出 2 行，cal_total_num = 144 // 2”；  
  - 外层按 `cal_idx` 遍历 H-block，`is_padding_row` 在首块/尾块使用不同模式补零或边缘复制。
- `is_padding_row` + “首/中/尾 三段循环”就是对 H 方向做 tiling + 边界处理。

**W 方向：**

- 阵列一次计算 32 宽或 64 宽甚至 128 宽：
  - 如 `4(h) × 32(w) × 8(oc)`、`1(h) × 64(w) × 16(oc)`、`1(h) × 128(w) × 8(oc)` 等。
- 对于宽度 256，例如 PDF 里明确写：  
  - “大小为（256,144）+ concat 一，先计算完一组（128,144）的两层，再计算下一组（128,144）。”
- 即：**W 方向先切成若干 128 宽的 tile**，每个 tile 内部再按照 32/64 宽度的阵列模板去排布与存储。

### 3.3 通道 tiling：Cin 与 Cout

**Cin：**

- 以 4ic 或 8ic 为 group。
- DataLoader 的 `line_buffer_reshape` + `transnum` 控制：
  - 每个 group 内，一次加载几行、每 cycle 灌多少个 channel。
- WeightLoader 的 `is_new` 控制：
  - group 的第一轮覆盖 acc_reg (`is_new = 0`)；
  - 随后的 group 不断 `is_new = 1` 累加到同一个 acc_reg，最后得到全 Cin 的结果。

**Cout：**

- 以 8/16/32 oc 为 group。
- 通过：
  - 阵列的上下/左右分块（前 4oc / 后 4oc；前 8oc / 后 8oc；4 组 8oc 等）；
  - WeightLoader 中 `weight_parall_mode`（每 cycle 8/16/32 个权重）；
  - DataStorer 的 `acc_mode`（控制从多个 acc_reg 组合结果、重排顺序）  
  实现对 Cout 的 tiling。

---

## 4. 标准化的“模板”：line_buffer 模式 / padding 模式 / store 模式

这些模式在《指令.xlsx》和 PDF 里反复出现，后端 codegen 就是围绕它们写循环。

### 4.1 line_buffer 加载模式（`line_buffer_reshape`）

常见几种（示意）：

- **模式 0**：连续 load 多行，直接填入 line_buffer（每行连续，按 H→W 填），适合绝大部分 standard conv。
- **模式 1**：前 2cycle 填四行的前 64，后 2cycle 填后 64，适合需要对 128 宽分块处理的情况。
- **模式 2 / 3**：每 cycle 填四行，每行 8 或 32 点，适合 grouped conv 或 deformable 输入重排。

### 4.2 行 padding 模式（`is_padding_row`）

典型枚举意义：

- `0`：不补零
- `1`：第一次写入 line_buffer 全 0（顶端 padding）
- `2`：`transnum` 次写入 line_buffer 全 0
- `3`：第一次写入前一半为 0
- `4`：前两次写入全 0
- `5`：最后一次写入全 0 / 后半为 0
- `6/7`：尾部更多变种
- 后续版本加 `8~11`：前/后几次写入做“边缘复制”而不是补 0

### 4.3 行内 padding / 特殊模式

在 deformable 或 bicubic 场景中：

- “行padding模式一/二/三/四”：  
  - 一般约定：
    - 模式一：整行填 0；
    - 模式二：只在第一行或最后一行的前/后 32 个 reg 进行 padding；
    - 模式三/四：配合 line_buffer 加载模式做“复制边界行”、“复制两行生成四行”等。

### 4.4 weight 部署方式（`weight_parall_mode` + `is_padding_col`）

在 PDF 里出现：

- “weight 部署方式二：一次 load 8 个 8bit 有效权重，line_buffer 切换顺序为 kx, 再 ic。”
- “weight 部署方式三：一次 load 16 个有效 8bit 权重，line_buffer 切换顺序为 kx, ic 方向先行内然后换行。”

这些都可以归结为：

- `weight_parall_mode` 决定“每 cycle load 8/16/32 个权重”；
- `is_padding_col` 决定在一行权重向量前后补多少 0；
- `line_buffer_row_shift` 决定权重对应的输入行如何在 double line buffer 上滚动。

### 4.5 store 模式 & acc_mode

在 DataStorer 中：

- `store_mode`：
  - “存 4cycle 数据 / 存 8cycle 数据 / 存 64cycle / 存 128cycle / 存 2cycle”等；
- `acc_mode`：
  - 决定在 acc_reg 里 0~7 这些 index 以何种顺序加权：
    - 1/2 平均：`(0+1)/2`, `(2+3)/2`...
    - 1/4 平均：`(0+1+2+3)/4`...
    - 1/8 + 特别跳取：bicubic 场景；
- `pooling_out_mode`：
  - 控制 4×4 / 8×8 pooling 如何从多次阵列输出组合结果。

---

## 5. 每层 tiling 模板（UNet 主干）

PDF 中按大小 `(H, W)` 分段给了典型算例，代码 `sd_codegen.py` 的 layer-by-layer 也是围绕这些模板写死的。下面用“模板”的方式梳理，让读者能根据 `(H,W,Cin,Cout)` 找到匹配的模式。

> 说明：下面只是抽象出关键 idea，具体数值（比如 `bas_addr` 增量）可以到 `sd_codegen.py` 里看对应实现。

### 5.1 模板 A：输入 256×144，conv3×3，C1→C4（`layer 0`）

**空间 tiling：**

- 整幅宽 256 × 高 144。
- 在 W 方向拆成 **左 128×144 + 右 128×144 两个大 tile**：
  - 左 tile：`bas_addr_cur = 0`，输出地址从 0 开始。
  - 右 tile：`bas_addr_cur = 288`（因为每行 256，按 4×addr/line 计算），输出地址从 `144*4` 开始。
- 在 H 方向：
  - 每次算 2 行：`cal_total_num = 144 // 2`；
  - 对于每一次 cal：
    - 上一组 cal（顶端）：`is_padding_row = 1`；
    - 中间：`0`；
    - 底部：`5`（尾部 padding）。

**通道 tiling：**

- Cin=1 → Cout=4，groups=1。
- 一次性处理所有 Cin（1 个），不需要 ic 分组。
- Cout 通过 WeightLoader 在 acc_reg 的两个半阵列上分别算“前 2 oc / 后 2 oc”，最后在 DataStorer 里按连续 oc 顺序输出。

**相关指令：**

- `QuantLoader`：加载该层的 scale/zp，`transnum=4`。
- DataLoader（每个 tile）：
  - `transnum=4`，`line_buffer_reshape=0`（每次 load4 行，直接填 line_buffer）。
  - `src_buffer_idx='offchip_input_buffer'`。
  - `bas_addr` 从 0 递增，首块步长 2（考虑 padding），之后 4。
- WeightLoader：
  - `weight_parall_mode=0`（每 cycle 8 个权重），`transnum=9`（3×3×Cin 的权重数 / 并行因子）。
  - `is_new=0`（因为 Cin=1，没有多 group 累加）。
  - `bas_addr` 固定，用完再整体 `+9` 到下一层。
- DataStorer：
  - `dest_buffer_idx='a'`，`store_mode=0`（每次写固定的 4cycle），`stride=144`。
  - H 方向以 2 行为步进，W=128 时总共写 `144*4*2` 个位置。

**在 `sd_codegen.py` 中的位置：**

- `# layer 0  256*144 conv1` 后面第一大段 for 循环（左半）、第二段 for 循环（右半）就是这个模板的具体实现。

---

### 5.2 模板 B：输入 256×144，conv3×3，C4→C4（`layer 1`）

**空间 tiling：**

- 同样拆成左 128×144 + 右 128×144。
- H 方向依旧 `cal_total_num = 144 // 2`，每次算 2 行。

**通道 tiling：**

- Cin=4 → Cout=4，groups=1。
- 这里出现 **ic 分组**：
  - 定义 `load_num_per_cal = 4`，表示 **4 个 ic group**；
  - 外层 `for cal_idx in range(cal_total_num)` 遍历 H-block，
  - 内层 `for load_per_cal_index in range(load_num_per_cal)` 遍历每个 group 的 ic。
- DataLoader：
  - `bas_addr = base + 144 * load_per_cal_index`，在同一空间块上轮流从不同 ic 组读。
- WeightLoader：
  - `is_new = 0 if load_per_cal_index == 0 else 1`：
    - 第一组 ic：覆盖 acc_reg；
    - 后三组 ic：累加到同一 acc_reg。
- DataStorer：
  - 每处理完 4 组 ic，才调用一次 DataStorer，把本次 2 行 × 128 宽 × Cout 的结果写入 buffer b。

**在 `sd_codegen.py` 中的位置：**

- `# layer 1  256*144 conv1_1`，可以看到上面描述的双层 for 循环与 `is_new` 控制。

同理：

- `layer 2` 增加了 pooling（所以 DataStorer 会设置 `is_pooling=1 / pooling_out_mode=...`），并为 concat 预留 buffer 区间。
- 中间层尺寸缩小（例如 `(64,36)`、`(32,18)`、`(16,9)` 等）时，H/W tiling 与 Cin/Cout 分组类似，但 `cal_total_num / load_num_per_cal / stride / store_mode` 等参数会变化，以匹配更小的 feature map 与不同的 pooling / pixelshuffle。

---

### 5.3 其它典型模板关键词（参考 PDF 和代码）

在 PDF 和 `sd_codegen.py` 的后续层里，可以按“章节名 + case 名字”对应：

- `三、大小为（64, 36）` → 部分 downsample/pool 后的中间层：
  - H=36，W=64（或平铺为 tile(8,16,3) 这类）。
  - 常见模式：
    - “一次从 a 中 load4 个 cycle … line_buffer 中直接存储，排布 4<-2ic<-64”；
    - `weight 部署方式三：一次 load16 个有效权重`。
- `四、大小为（32, 18）`、`五、大小为（16, 9）`：
  - 进一步下采样后的层，H/W 都是较小整数，基本复用相同 line_buffer 模式与 weight 模式，只是 `cal_total_num` 和 address 步长不同。
- `八、大小为（128，72） + concat 二`、`九、大小为（256，144） + concat 一`：
  - 涉及 concat 的层，会：
    - 先计算其中一部分，输出到 buffer a/b；
    - 再通过 DataLoader 的 `bas_addr = 144*4*2 + 72*8` 等偏移，把前一段结果与新结果在 H/W or C 方向 concat 到一起；
    - 为了后续层方便，DataStorer 会设置不同的 `dest_buffer_idx` 和 `base_addrs_res_cur_for_cat`。

**对于每一个这样的章节：**

- PDF 先给出：“一次输入从 a/b 中 load 几个 cycle … 多少 cycle 算完 … 输出尺寸多少”；
- 然后给出清晰的：
  - dataloader: `bas_addr`, `src_buffer_idx`, `transnum`, `is_padding_row`, `line_buffer_reshape`；
  - weightloader: `transnum`, `weight_parall_mode`, `is_padding_col`, `line_buffer_row_shift`, `kernal_size`；
  - datastorer: `dest_buffer_idx`, `store_mode`, `acc_mode`, `transfer_num`, `quant_mode`, `pooling_out_mode` 或 `pixelshuffle_mode_out`。

`sd_codegen.py` 就是照着这些表格，把这些数值和 for 循环、address 累加公式写成 Python。

---

## 6. FSRCNN / SR pipeline 的 tiling（`sd_sr_codegen.py`）

`sd_sr_codegen.py` 的结构与 `sd_codegen.py` 类似，只是：

- 针对 **FSRCNN + UNet offset + bicubic 插值** 场景；
- 需要考虑 **UNet 已经占用的 weight/quant buffer 空间**，所以：
  - `WeightLoaderManager.bas_addr_cur = [1737, 792, 1152]`；
  - `QuantLoaderManager.bas_addr_cur = 665`；
- 多了 `OffsetLoaderManager`，用来加载 offset 权重与 offset map。

`sd_sr_codegen.py` 中的 `sd_inst(is_first=False, load_next=True)`：

1. 初始化各 manager 的 bas_addr。
2. 若 `is_first`，先 offchip 加载整套 Unet 的 quant/weight（共享）。
3. 为 SR 的每一层：
   - 照着 PDF 中 “(32,4,3)”、“(8,16,3)”、“(32,8,1)”、“(8,4,3)” 等 case：
     - 用 DataLoader 做行/列 tiling + padding；
     - 用 WeightLoader 做 ic/oc 分组；
     - 需要 offset 的层，用 OffsetLoader/WeightLoader 的 `is_bilinear_bicubic=1/2`；
     - 用 DataStorer 做 pixelshuffle / pooling / mask 输出；
   - 同样通过 magic number 的 `bas_addr` 计算来处理 concat 与多阶段上采样。

---

## 7. 三个代码文件的职责与“写死点”

### 7.1 `instruction.py` —— **硬件 ISA 封装（通用）**

- 抽象了 7 类指令。
- 负责：
  - 统一的 `dispatch()` API；
  - 为每条指令分配一个自增 `code_num`；
  - 把生成的指令 append 到 `Inst.code_list`。
- **不承担：**
  - 模型结构；
  - tiling 策略；
  - 地址规划；  
  这些都由上层脚本决定。

### 7.2 `sd_codegen.py` —— **UNet 的专用 codegen + 调度**

写死的东西包括：

- **UNet 结构：**`layer_config` 列表中显式给出每层的 Cin/Cout/kernel/groups。
- **输入分辨率：**默认假定 256×144。
- **每层的 tiling 模板：**
  - 是否拆左右 128×144；
  - 每次算几行（`cal_total_num = H//2` 等）；
  - padding 行数 `padding_num`；
  - `transnum`、`line_buffer_reshape`、`weight_parall_mode`、`acc_mode`、`store_mode` 等；
  - quant/weight 的加载顺序和 `bas_addr` 起点。
- **地址/offset 公式：**
  - 大量形如 `144*4*2+72*8` 的常数，用于 concat 与多层输出在 buffer 中的布局。
- **简单的 buffer 模型：**
  - `buffer_a_model` / `buffer_b_model` 用字典记录某段地址存的是哪一层输出、是否仍然有效。

### 7.3 `sd_sr_codegen.py` —— **FSRCNN+SR 的专用 codegen + 调度**

类似地写死：

- SR 结构和各层尺寸（如 `(32,4,3)`、`(8,16,3)` 等）；
- Unet 已经使用过的 weight/quant buffer 区间（通过 bas_addr 初值体现）；
- 各层的 tiling / pixelshuffle / bilinear/bicubic 插值的实现细节。

---

## 8. 面向“通用前端”的重构建议

你想做的是：**PyTorch/ONNX → Relay →（通用）前端 →（模板化）后端 → Instructions**，而不是继续在 `sd_codegen.py` 里为每个模型手写。

建议按三层拆分：

### 8.1 前端：从 Relay 抽象出 Layer 描述（不带 tiling）

定义一个统一的 Layer 描述结构，例如：

```python
class LayerDesc:
    op: str          # "conv2d" / "pool2d" / "upsample" / "pixelshuffle" / "concat" ...
    idx: int
    H_in: int
    W_in: int
    Cin: int
    Cout: int
    kH: int
    kW: int
    stride_h: int
    stride_w: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int
    groups: int
    deformable: bool
    need_pixelshuffle: bool
    need_mask: bool
    # ...
```

从 Relay IR 自动遍历生成 `List[LayerDesc]`，**这是前端需要做的工作**，完全不涉及：

- `is_padding_row` / `transnum`；
- `bas_addr` / `stride`；
- 任何寄存器或 buffer 细节。

### 8.2 中间：tiling 策略 & 模板选择

在一个新的“后端库”中，把 PDF/现有代码里的模式封装成有限个模板，例如：

```python
class TilingCfg:
    H_tile: int       # 每 tile 高度，如 4 或 6
    W_tile: int       # 每 tile 宽度，如 32/64/128
    Cin_group: int    # 每次累加的 ic 数，如 4/8
    Cout_group: int   # 每次计算的 oc 数，如 8/16
    line_buffer_mode: int
    padding_mode_row_head: int
    padding_mode_row_body: int
    padding_mode_row_tail: int
    weight_mode: int
    store_mode: int
    acc_mode: int
    pixelshuffle_mode: Optional[int]
    pooling_mode: Optional[int]
    # ...
```

- 为每类常见场景（`(256,144)`、`(128,72)`、`(64,36)`、`(32,18)`、`(16,9)`、各类 concat / upsample）定义一个或几个 **Tiling 模板**，内部规则和现有手写代码等价。
- 写一个函数：

```python
def choose_tiling(desc: LayerDesc) -> List[TilingCfg]:
    # 根据 H/W/Cin/Cout/op/group/stride 等，返回一个或多个 tile 配置
```

这里可以沿用现在的策略：比如只支持特定输入尺寸、固定的 H/W 模板等等，但**决策逻辑集中在一个地方**。

### 8.3 后端：模板 → 微指令（使用 `instruction.py`）

为每个模板写统一的“微指令生成器”，例如：

```python
def emit_conv3x3(layer: LayerDesc, tiling: TilingCfg, managers: Managers):
    # 1. 根据 tiling.H_tile/W_tile 计算 cal_total_num/load_total_num
    # 2. 计算每个 tile 的 bas_addr / base_addrs_res
    # 3. 用 DataLoader/WeightLoader/QuantLoader/DataStorer.dispatch 生成指令
```

这样：

- **写死的**：
  - 指令格式（ISA）；
  - 少量 tiling 模板（与现在 pdf/代码中现有模式一致）；
  - 地址计算公式。
- **可扩展的**：
  - 前端只要对新模型生成 `LayerDesc`，不需要修改后端模板；
  - 只要新模型的尺寸落在支持的模板集合里，后端自然能生成正确的指令。
  - 将来增加新硬件特性（更多 tile 模式 / buffer），只需要在模板库中扩展即可。

---

## 9. 总结

- **Excel《指令.xlsx》**：给出了每类微指令的字段与取值语义，是硬件 ISA 的 ground truth。
- **PDF《计算过程》**：在各个输入大小 `(H,W)` 下，手推了具体的 tiling / line_buffer 模式 / weight 部署 / store 模式，并给了详尽的例子。
- **`instruction.py`**：把指令集封装成 Python API，是完全通用的后端工具。
- **`sd_codegen.py` / `sd_sr_codegen.py`**：针对某个固定 UNet / FSRCNN+SR 网络，把 PDF 中的计算过程全部“翻译”为 for 循环 + magic number，属于**模型专用的后端 codegen + 调度**，在这里写死了模型结构、输入尺寸和 tiling 策略。

如果要支撑 **通用前端（Relay → Instructions）**：

- 前端应**只描述算子语义与 shape（LayerDesc）**；
- 所有与行/列/通道 tiling、buffer 地址、line_buffer/weight/store 模式有关的细节，统一放在后端的 **模板库 + 地址计算公式** 中，用少数几个 `emit_xxx` 函数去生成微指令；
- 这样，开发者只需要阅读这一份文档和这些模板实现，就可以理解“每一层如何 tiling、对应的微指令长什么样”。