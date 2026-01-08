import ast
import sys
import os
from collections import OrderedDict
from typing import List

# 配置位宽
FIELD_WIDTHS = {
    "OffchipDataLoader": {
        "transnum": 12,
        "load_model": 2,
        "src_buffer_idx": 2,
        "bas_addr": 12,
    },
    "DataLoader": {
        "layer_idx": 5,
        "line_buffer_reshape": 3,
        "is_padding_row": 4,
        "read_mode": 1,
        "transnum": 4,
        "line_buffer_idx": 1,
        "src_buffer_idx": 2,
        "bas_addr": 12,
    },
    "WeightLoader": {
        "acc_reg_comp_idx": 1,
        "kernal_size": 1,
        "line_buffer_row_shift": 3,
        "line_buffer_idx": 1,
        "is_padding_col": 3,
        "weight_parall_mode": 2,
        "is_new": 1,
        "transnum": 6,
        "is_bilinear_bicubic": 2,
        "offset_reg_idx": 1,
        "bas_addr": 12,
    },
    "OffsetLoader": {
        "offset_reg_idx": 1,
        "bas_addr": 8,
    },
    "QuantLoader": {
        "quant_reg_load_idx": 1,
        "quant_mode": 3,
        "layer_idx": 5,
        "transnum": 6,
        "bas_addr": 10,
    },
    "DataStorer": {
        "quant_config_idx": 1,
        "pixelshuffle_out_mode": 2,
        "is_pixelshuffle": 1,
        "pooling_out_mode": 3,
        "pooling_out_new": 1,
        "is_pooling": 1,
        "reg_out_idx": 1,
        "acc_mode": 3,
        "transfer_num": 2,
        "store_mode": 2,
        "stride": 8,
        "is_bicubic_add": 1,
        "is_first_or_last_row": 2,
        "is_mask": 1,
        "is_new": 1,
        "dest_buffer_idx": 3,
        "base_addr_pooling": 12,
        "base_addrs_res": 12,
    },
    "OffchipDataStorer": {
        "src_buffer": 1,
        "transnum": 12,
        "base_addr": 12,
    },
}

# 固定位宽
OPCODE_WIDTH = 3
SRC_DEST_WIDTH = 4

# 忽略字段
IGNORE_KEYS = {"code_num", "dependency"}

# 最终追加顺序（低位）
TAIL_ORDER = ["src1", "src2", "src3", "src4", "dest", "op_code"]

# op_code 映射
OPCODE_MAP = {
    "OffchipDataLoader": 0,
    "DataLoader": 1,
    "WeightLoader": 2,
    "OffsetLoader": 3,
    "QuantLoader": 4,
    "DataStorer": 5,
    "OffchipDataStorer": 6,
}

# 映射表（字符串到索引）
DATA_LOADER_SRC_BUFFER = {"a": 0, "b": 1, "offchip_input_buffer": 2}
DATA_STORER_DEST_BUFFER = {
    "a": 0,
    "b": 1,
    "fsrcnn_output_buffer": 2,
    "unet_output_reg": 3,
    "offset_reg": 4,
}
OFFCHIP_STORER_SRC = {"unet_output_reg": 0, "fsrcnn_output_buffer": 1}
SPECIAL_TRANSMAP = {"unet_total": 2048}

# -------------------- 辅助函数 --------------------
def parse_line_to_ordered_dict(line: str) -> OrderedDict:
    obj = ast.literal_eval(line.strip())
    if not isinstance(obj, dict):
        raise ValueError("每行必须是 dict 字面量")
    return OrderedDict(obj)


def normalize_value(key, raw_val, opname):
    """将 raw_val 规范化为整数，包含特殊字符串映射"""
    # list -> first
    if isinstance(raw_val, list):
        raw_val = raw_val[0] if raw_val else 0
    if isinstance(raw_val, bool):
        return 1 if raw_val else 0
    if isinstance(raw_val, (int,)):
        return int(raw_val)
    if isinstance(raw_val, float):
        return int(raw_val)
    if isinstance(raw_val, str):
        s = raw_val.strip()
        # op_code 字段：可能是名称
        if key == "op_code":
            # map by OPCODE_MAP, otherwise if digit parse
            if s in OPCODE_MAP:
                return OPCODE_MAP[s]
            if s.isdigit():
                return int(s)
            # ignore-case match
            for k, v in OPCODE_MAP.items():
                if k.lower() == s.lower():
                    return v
            raise ValueError(f"未知 op_code 字符串: '{s}'")
        # transnum special for OffchipDataLoader
        if key == "transnum":
            if s in SPECIAL_TRANSMAP:
                return SPECIAL_TRANSMAP[s]
            if s.isdigit():
                return int(s)
            raise ValueError(f"无法解析 transnum 字符串: '{s}'")
        if key == "src_buffer_idx":
            if s in DATA_LOADER_SRC_BUFFER:
                return DATA_LOADER_SRC_BUFFER[s]
            if s.isdigit():
                return int(s)
            raise ValueError(f"无法解析 src_buffer_idx: '{s}'")
        if key == "dest_buffer_idx":
            if s in DATA_STORER_DEST_BUFFER:
                return DATA_STORER_DEST_BUFFER[s]
            if s.isdigit():
                return int(s)
            raise ValueError(f"无法解析 dest_buffer_idx: '{s}'")
        if key == "src_buffer":
            if s in OFFCHIP_STORER_SRC:
                return OFFCHIP_STORER_SRC[s]
            if s.isdigit():
                return int(s)
            raise ValueError(f"无法解析 OffchipDataStorer.src_buffer: '{s}'")
        # generic numeric string
        if s.isdigit():
            return int(s)
        if s.lower() in ("true", "false"):
            return 1 if s.lower() == "true" else 0
        raise ValueError(f"无法解析字段 {key} 的字符串值: '{s}'")
    raise ValueError(f"不支持的字段类型: {type(raw_val)} for key {key}")


def int_to_bin(value: int, width: int) -> str:
    if value < 0:
        raise ValueError("不支持负数")
    b = format(int(value), "b")
    if len(b) > width:
        raise ValueError(f"value {value} 无法用 {width} 位表示 (需要 {len(b)} 位)")
    return b.zfill(width) # 左侧填0补足位宽


# -------------------- 编码单条指令 --------------------
def encode_instruction(record, split) -> str:
    # op_code 必须存在
    if "op_code" not in record:
        raise ValueError("记录缺少 op_code 字段: " + str(record))
    opname_raw = record["op_code"]
    opcode_val = normalize_value("op_code", opname_raw, opname_raw)

    # 获取该 opcode 的字段位宽表
    opname_key = opname_raw if isinstance(opname_raw, str) else None
    if isinstance(opname_raw, str):
        # try direct name; also accept names with 'Ins' suffix by stripping if needed
        opname_candidate = opname_raw
        if opname_candidate not in FIELD_WIDTHS and opname_candidate.endswith("Ins"):
            opname_candidate = opname_candidate[:-3]
        opname = opname_candidate
    else:
        opname = None

    if opname not in FIELD_WIDTHS:
        raise ValueError(f"未在 FIELD_WIDTHS 中为 opcode '{opname}' 指定位宽。请在脚本顶部添加 FIELD_WIDTHS['{opname}']。")

    width_map = FIELD_WIDTHS[opname]

    # extra fields (原顺序的逆序)，排除 ignore 和 tail (src/dest/op_code)
    extra_fields = [k for k in reversed(record.keys()) if k not in IGNORE_KEYS and k not in TAIL_ORDER]

    bit_parts = []
    # 处理 extra fields（从高位到低位）
    for k in extra_fields:
        if k not in width_map:
            raise ValueError(f"Opcode '{opname}' 的字段 '{k}' 未在 FIELD_WIDTHS['{opname}'] 中指定位宽")
        raw_val = record[k]
        v = normalize_value(k, raw_val, opname_raw)
        w = width_map[k]
        b = int_to_bin(v, w)
        bit_parts.append(b)
        if split:
            bit_parts.append(" ")

    # 处理 tail 部分（低位），固定宽度： src4..src1,dest(4), op_code(3)
    tail_bits = []
    # src4..src1 & dest: width SRC_DEST_WIDTH each
    for t in ["src1", "src2", "src3", "src4", "dest"]:
        raw = record.get(t, 0)
        # some fields could be named dest_buffer_idx instead of dest in certain opcodes: handle before normalization
        # but per spec dest/src1..4 are numeric indices in the record; we accept string numeric or int
        v = normalize_value(t, raw, opname_raw) if raw is not None else 0
        tail_bits.append(int_to_bin(v, SRC_DEST_WIDTH))
        if split:
            tail_bits.append(" ")

    # opcode
    tail_bits.append(int_to_bin(opcode_val, OPCODE_WIDTH))

    full_bitstr = "".join(bit_parts) + "".join(tail_bits)
    return full_bitstr


def split_into_32_chunks(out_lines: List[str]) -> List[str]:
    result: List[str] = []

    for idx, bitstr in enumerate(out_lines):
        if bitstr is None:
            continue
        # 清理空白并验证
        s = bitstr.strip()
        if s == "":
            continue
        # 可选：校验只含 0/1
        if any(c not in "01" for c in s):
            raise ValueError(f"第 {idx} 条二进制码包含非 0/1 字符: {s!r}")

        # 计算补齐长度（向左补零）
        rem = len(s) % 32
        if rem != 0:
            pad = 32 - rem
            s_padded = ("0" * pad) + s
        else:
            s_padded = s

        # 从低位（右边）开始每 32 位一块切割
        # 例如: s_padded = [ ... highest ... | mid | lowest ]
        # 我们需要按顺序返回 [lowest, mid, highest]
        n = len(s_padded) // 32
        for i in range(n):
            # chunk i: the i-th lowest 32-bit block
            # compute slice indices:
            # right_end = len - 32*i
            right_end = len(s_padded) - 32 * i
            left_start = right_end - 32
            chunk = s_padded[left_start:right_end]
            result.append(chunk)

    return result



# -------------------- 主流程 --------------------
def compile_file(input_path, output_path, split, pad_and_cut):
    with open(input_path, "r", encoding="utf8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    out_lines = []
    for idx, line in enumerate(lines):
        try:
            rec = parse_line_to_ordered_dict(line)
            bitstr = encode_instruction(rec, split)
            out_lines.append(bitstr)
        except Exception as e:
            print(f"[Error] 第 {idx+1} 行 处理失败: {e}")
            raise
    
    # 处理为可放入指令队列的格式
    if (not split) and pad_and_cut:
        out_lines = split_into_32_chunks(out_lines)

    # 写输出
    with open(output_path, "w", encoding="utf8") as fo:
        for b in out_lines:
            fo.write(b + "\n")

    print("输出写入:", output_path)


# -------------------- CLI --------------------
if __name__ == "__main__":
    infile = "code/conv_test_data/conv_test_ins.txt"
    outfile = "code/conv_test_data/conv_test_ins_assembly_split.txt"
    split = True
    pad_and_cut = False
    compile_file(infile, outfile, split, pad_and_cut)
