import copy
from typing import List, Dict

class Inst:
    current_code_num = 0
    code_list: List[Dict] = []


class OffchipDataLoader:
    @staticmethod
    def dispatch(transnum, load_model, src_buffer_idx, bas_addr):
        code = {
            "code_num": [Inst.current_code_num],
            "op_code": "OffchipDataLoader",
            "transnum": transnum,
            "load_model": load_model,
            "src_buffer_idx": src_buffer_idx,
            "bas_addr": bas_addr
        }
        Inst.current_code_num += 1
        Inst.code_list.append(code)
        return code

class DataLoader:
    @staticmethod
    def dispatch(layer_idx, line_buffer_reshape, is_padding_row, read_mode, transnum, line_buffer_idx, src_buffer_idx, bas_addr):
        code = {
            "code_num": [Inst.current_code_num],
            "op_code": "DataLoader",
            "layer_idx": layer_idx,
            "line_buffer_reshape": line_buffer_reshape,
            "is_padding_row": is_padding_row,
            "read_mode": read_mode,
            "transnum": transnum,
            "line_buffer_idx": line_buffer_idx,
            "src_buffer_idx": src_buffer_idx,
            "bas_addr": bas_addr
        }
        Inst.current_code_num += 1
        Inst.code_list.append(code)
        return code



class WeightLoader:
    @staticmethod
    def dispatch(acc_reg_comp_idx, kernal_size, line_buffer_row_shift, line_buffer_idx, is_padding_col, weight_parall_mode, 
                is_new, transnum, bas_addr, is_bilinear_bicubic, offset_reg_idx):
        code = {
            "code_num": [Inst.current_code_num],
            "op_code": "WeightLoader",
            "acc_reg_comp_idx": acc_reg_comp_idx,
            "kernal_size": kernal_size,
            "line_buffer_row_shift": line_buffer_row_shift,
            "line_buffer_idx": line_buffer_idx,
            "is_padding_col": is_padding_col,
            "weight_parall_mode": weight_parall_mode,
            "is_new": is_new,
            "transnum": transnum,
            "is_bilinear_bicubic": is_bilinear_bicubic,
            "offset_reg_idx": offset_reg_idx,
            "bas_addr": bas_addr
        }
        Inst.current_code_num += 1
        Inst.code_list.append(code)
        return code


class OffsetLoader:
    @staticmethod
    def dispatch(offset_reg_idx, bas_addr):
        code = {
            "code_num": [Inst.current_code_num],
            "op_code": "OffsetLoader",
            "offset_reg_idx": offset_reg_idx,
            "bas_addr": bas_addr
        }
        Inst.current_code_num += 1
        Inst.code_list.append(code)
        return code


class QuantLoader:
    @staticmethod
    def dispatch(quant_reg_load_idx, quant_mode, layer_idx, transnum, bas_addr):
        code = {
            "code_num": [Inst.current_code_num],
            "op_code": "QuantLoader",
            "quant_reg_load_idx": quant_reg_load_idx,
            "quant_mode": quant_mode,
            "layer_idx": layer_idx,
            "transnum": transnum,
            "bas_addr": bas_addr,
        }
        Inst.current_code_num += 1
        Inst.code_list.append(code)
        return code


class DataStorer:
    @staticmethod
    def dispatch(quant_config_idx, pixelshuffle_out_mode, is_pixelshuffle, pooling_out_mode, pooling_out_new, is_pooling, reg_out_idx, acc_mode,
                transfer_num, store_mode, stride, base_addr_pooling, base_addrs_res, is_bicubic_add, is_first_or_last_row, is_mask, is_new, dest_buffer_idx):
        code = {
            "code_num": [Inst.current_code_num],
            "op_code": "DataStorer",
            "pooling_out_new": pooling_out_new,
            "quant_config_idx": quant_config_idx,
            "pixelshuffle_out_mode": pixelshuffle_out_mode,
            "is_pixelshuffle": is_pixelshuffle,
            "pooling_out_mode": pooling_out_mode,
            "is_pooling": is_pooling,
            "reg_out_idx": reg_out_idx,
            "acc_mode": acc_mode,
            "transfer_num": transfer_num, # 这里不代表具体个数，而是cycle
            "store_mode": store_mode,
            "stride": stride,
            "is_bicubic_add": is_bicubic_add,
            "is_first_or_last_row": is_first_or_last_row,
            "is_mask": is_mask,
            "is_new": is_new,
            "dest_buffer_idx": dest_buffer_idx,
            "base_addr_pooling": base_addr_pooling,
            "base_addrs_res": base_addrs_res
        }
        Inst.current_code_num += 1
        Inst.code_list.append(code)
        return code

class OffchipDataStorer:
    @staticmethod
    def dispatch(src_buffer, transnum, base_addr):
        code = {
            "code_num": [Inst.current_code_num],
            "op_code": "OffchipDataStorer",
            "src_buffer": src_buffer,
            "transnum": transnum,
            "base_addr": base_addr
        }
        Inst.current_code_num += 1
        Inst.code_list.append(code)
        return code
