from instruction import *
import math
import numpy as np
import time
import copy
from typing import Union
import datetime

layer_config = [
    {"layer_index": 0, "c_in": 1, "c_out": 4, "kernel_size": 3, "groups": 1},
    {"layer_index": 1, "c_in": 4, "c_out": 4, "kernel_size": 3, "groups": 1},
    {"layer_index": 2, "c_in": 4, "c_out": 4, "kernel_size": 3, "groups": 1},
    {"layer_index": 3, "c_in": 4, "c_out": 8, "kernel_size": 3, "groups": 1},
    {"layer_index": 4, "c_in": 8, "c_out": 8, "kernel_size": 3, "groups": 1},
    {"layer_index": 5, "c_in": 8, "c_out": 16, "kernel_size": 3, "groups": 1},
    {"layer_index": 6, "c_in": 16, "c_out": 16, "kernel_size": 3, "groups": 1},
    {"layer_index": 7, "c_in": 16, "c_out": 64, "kernel_size": 3, "groups": 2},
    {"layer_index": 8, "c_in": 64, "c_out": 64, "kernel_size": 3, "groups": 8},
    {"layer_index": 9, "c_in": 64, "c_out": 64, "kernel_size": 3, "groups": 8},
    {"layer_index": 10, "c_in": 64, "c_out": 256, "kernel_size": 3, "groups": 8},
    {"layer_index": 11, "c_in": 128, "c_out": 16, "kernel_size": 3, "groups": 2},
    {"layer_index": 12, "c_in": 16, "c_out": 64, "kernel_size": 3, "groups": 2},
    {"layer_index": 13, "c_in": 32, "c_out": 16, "kernel_size": 3, "groups": 1},
    {"layer_index": 14, "c_in": 16, "c_out": 32, "kernel_size": 3, "groups": 1},
    {"layer_index": 15, "c_in": 16, "c_out": 8, "kernel_size": 3, "groups": 1},
    {"layer_index": 16, "c_in": 8, "c_out": 16, "kernel_size": 3, "groups": 1},
    {"layer_index": 17, "c_in": 8, "c_out": 4, "kernel_size": 3, "groups": 1},
    {"layer_index": 18, "c_in": 4, "c_out": 1, "kernel_size": 3, "groups": 1}
]

class DataLoaderManager:
    def __init__(self):
        self.line_buffer_load_total_num = 0
        self.src_buffer_idx_on_chip = 'a' # a or b
        self.line_buffer_idx = 0 # 0 or 1
        self.layer_idx = 0

    def src_buffer_idx_on_chip_switch(self):
        self.src_buffer_idx_on_chip = 'b' if self.src_buffer_idx_on_chip == 'a' else 'a'

    def line_buffer_idx_switch(self):
        self.line_buffer_idx = 1 if self.line_buffer_idx == 0 else 0


class WeightLoaderManager:
    def __init__(self):
        self.line_buffer_idx = 0 # 0 or 1
        self.acc_reg_idx = 0 # 0 or 1
        self.bas_addr_cur = [0, 0, 0] # 代表3种parall_mode所用的buffer

    def line_buffer_idx_switch(self):
        self.line_buffer_idx = 1 if self.line_buffer_idx == 0 else 0

    def acc_reg_idx_switch(self):
        self.acc_reg_idx = 1 if self.acc_reg_idx == 0 else 0

class QuantLoaderManager:
    def __init__(self):
        self.bas_addr_cur = 0 # 在过程中会不断累加上transnum，目前是scale和zp共用，假设是分开的两个buffer
        self.quant_config_idx = 0 # 0 or 1

    def quant_config_idx_switch(self):
        self.quant_config_idx = 1 if self.quant_config_idx == 0 else 0


class DataStorerManager:
    def __init__(self):
        self.dest_buffer_idx_on_chip = 'a' # a or b
        self.acc_reg_idx = 0 # 0 or 1
        self.quant_config_idx = 0 # 0 or 1
        self.base_addrs_res_cur = 0 # 表示buffer a 或 b 中此时要存的数据的起始地址
        self.base_addrs_res_cur_for_cat = 0 # 针对buffer a
        self.base_addr_pooling_cur = 0 # 针对buffer a
        self.buffer_a_model = [] # 简易建模，其中的填的是列表，如{"begin": n, "end": m, "info": 'layer1_out'}
        self.buffer_b_model = []

    def dest_buffer_idx_on_chip_switch(self):
        self.dest_buffer_idx_on_chip = 'b' if self.dest_buffer_idx_on_chip == 'a' else 'a'

    def acc_reg_idx_switch(self):
        self.acc_reg_idx = 1 if self.acc_reg_idx == 0 else 0

    def quant_config_idx_switch(self):
        self.quant_config_idx = 1 if self.quant_config_idx == 0 else 0

if __name__ == "__main__":
    dataloadermanager = DataLoaderManager()
    weightloadermanager = WeightLoaderManager()
    quantloadermanager = QuantLoaderManager()
    datastorermanager = DataStorerManager()

    # off load
    # quant
    OffchipDataLoader.dispatch(
        transnum = 'unet_total',
        load_model = 0,
        src_buffer_idx = 2,
        bas_addr = 0
    )
    OffchipDataLoader.dispatch(
        transnum = 'unet_total',
        load_model = 1,
        src_buffer_idx = 2,
        bas_addr = 0
    )
    # weight
    OffchipDataLoader.dispatch(
        transnum = 'unet_total',
        load_model = 0,
        src_buffer_idx = 1,
        bas_addr = 0
    )
    OffchipDataLoader.dispatch(
        transnum = 'unet_total',
        load_model = 1,
        src_buffer_idx = 1,
        bas_addr = 0
    )
    OffchipDataLoader.dispatch(
        transnum = 'unet_total',
        load_model = 2,
        src_buffer_idx = 1,
        bas_addr = 0
    )
    # image
    OffchipDataLoader.dispatch(
        transnum = 144*4,
        load_model = 0,
        src_buffer_idx = 0,
        bas_addr = 0
    )

    # layer by layer sim
    # layer 0  256*144 conv1
    layer_idx = 0
    quant_mode = 0
    QuantLoader.dispatch(
        quant_reg_load_idx = quantloadermanager.quant_config_idx,
        quant_mode = quant_mode,
        layer_idx = layer_idx,
        transnum = 4,
        bas_addr = quantloadermanager.bas_addr_cur
    )
    quantloadermanager.bas_addr_cur += 4

    # 左半部分128*144
    # 初始变量设定
    cal_total_num = 144//2 # 一次算出2行
    load_total_num = 144//2 
    dataloadermanager.line_buffer_load_total_num = load_total_num
    padding_num = 1
    dataloadermanager.bas_addr_cur = 0
    datastorermanager.base_addrs_res_cur = 0
    for load_idx in range(load_total_num):
        # DataLoader
        if load_idx < padding_num:
            is_padding_row = 1
        elif load_idx > load_total_num - 1 - padding_num:
            is_padding_row = 5
        else:
            is_padding_row = 0
        DataLoader.dispatch(
            layer_idx = layer_idx,
            line_buffer_reshape = 0,
            is_padding_row = is_padding_row,
            read_mode = 0,
            transnum = 4,
            line_buffer_idx = dataloadermanager.line_buffer_idx,
            src_buffer_idx = 'offchip_input_buffer',
            bas_addr = dataloadermanager.bas_addr_cur
        )
        dataloadermanager.line_buffer_idx_switch()
        dataloadermanager.bas_addr_cur += 2 if load_idx < padding_num else 4
        # WeightLoader
        WeightLoader.dispatch(
            acc_reg_comp_idx = weightloadermanager.acc_reg_idx,
            kernal_size = 0,
            line_buffer_row_shift = 1,
            line_buffer_idx = weightloadermanager.line_buffer_idx,
            is_padding_col = 1,
            weight_parall_mode = 0,
            is_new = 0, # 每次都覆盖
            transnum = 9,
            bas_addr = weightloadermanager.bas_addr_cur[0], # 每次用的都是同样的权重，bas_addr不需要递增
            is_bilinear_bicubic = 0,
            offset_reg_idx = 0
        )
        weightloadermanager.acc_reg_idx_switch()
        weightloadermanager.line_buffer_idx_switch()
        # DataStorer
        DataStorer.dispatch(
            quant_config_idx = datastorermanager.quant_config_idx,
            pixelshuffle_out_mode = 0,
            is_pixelshuffle = 0,
            pooling_out_mode = 0,
            is_pooling = 0,
            reg_out_idx = datastorermanager.acc_reg_idx,
            acc_mode = 0,
            transfer_num = 1,
            store_mode = 0,
            stride = 144,
            base_addr_pooling = 0,
            base_addrs_res = datastorermanager.base_addrs_res_cur,
            is_bicubic_add = 0,
            is_first_or_last_row = 0,
            is_new = 0,
            dest_buffer_idx = 'a' # 存到a
        )
        datastorermanager.acc_reg_idx_switch()
        datastorermanager.base_addrs_res_cur += 2

    # 右半部分128*144
    # 初始变量设定
    cal_total_num = 144//2
    load_total_num = 144//2 # 一次算出2行
    dataloadermanager.line_buffer_load_total_num = load_total_num
    padding_num = 1
    dataloadermanager.bas_addr_cur = 288
    datastorermanager.base_addrs_res_cur = 144*4
    
    for load_idx in range(load_total_num):
        # DataLoader
        if load_idx < padding_num:
            is_padding_row = 1
        elif load_idx > load_total_num - 1 - padding_num:
            is_padding_row = 5
        else:
            is_padding_row = 0
        DataLoader.dispatch(
            layer_idx = layer_idx,
            line_buffer_reshape = 0,
            is_padding_row = is_padding_row,
            read_mode = 0,
            transnum = 4,
            line_buffer_idx = dataloadermanager.line_buffer_idx,
            src_buffer_idx = 'offchip_input_buffer',
            bas_addr = dataloadermanager.bas_addr_cur
        )
        dataloadermanager.line_buffer_idx_switch()
        dataloadermanager.bas_addr_cur += 2 if load_idx < padding_num else 4
        # WeightLoader
        WeightLoader.dispatch(
            acc_reg_comp_idx = weightloadermanager.acc_reg_idx,
            kernal_size = 0,
            line_buffer_row_shift = 1,
            line_buffer_idx = weightloadermanager.line_buffer_idx,
            is_padding_col = 1,
            weight_parall_mode = 0,
            is_new = 0, # 每次都覆盖
            transnum = 9,
            bas_addr = weightloadermanager.bas_addr_cur[0], # 每次用的都是同样的权重，bas_addr不需要递增
            is_bilinear_bicubic = 0,
            offset_reg_idx = 0
        )
        weightloadermanager.acc_reg_idx_switch()
        weightloadermanager.line_buffer_idx_switch()
        # DataStorer
        DataStorer.dispatch(
            quant_config_idx = datastorermanager.quant_config_idx,
            pixelshuffle_out_mode = 0,
            is_pixelshuffle = 0,
            pooling_out_mode = 0,
            is_pooling = 0,
            reg_out_idx = datastorermanager.acc_reg_idx,
            acc_mode = 0,
            transfer_num = 1,
            store_mode = 0,
            stride = 144,
            base_addr_pooling = 0,
            base_addrs_res = datastorermanager.base_addrs_res_cur,
            is_bicubic_add = 0,
            is_first_or_last_row = 0,
            is_new = 0,
            dest_buffer_idx = 'a' # 存到a
        )
        datastorermanager.acc_reg_idx_switch()
        datastorermanager.base_addrs_res_cur += 2

    # 收尾
    weightloadermanager.bas_addr_cur[0] += 9 
    datastorermanager.buffer_a_model.append({"begin":0, "end":144*4*2, "info":"c1_layer0_out", "valid":True})
    quantloadermanager.quant_config_idx_switch()
    datastorermanager.quant_config_idx_switch()

    # layer 1  256*144 conv1_1
    layer_idx = 1
    quant_mode = 0
    QuantLoader.dispatch(
        quant_reg_load_idx = quantloadermanager.quant_config_idx,
        quant_mode = quant_mode,
        layer_idx = layer_idx,
        transnum = 4,
        bas_addr = quantloadermanager.bas_addr_cur
    )
    quantloadermanager.bas_addr_cur += 4

    # 左半部分128*144
    # 初始变量设定
    cal_total_num  = 144//2 # 一次算出2行
    load_num_per_cal = 4 # 遍历c_in
    load_total_num = cal_total_num * load_num_per_cal
    dataloadermanager.line_buffer_load_total_num = load_total_num
    padding_num = 1
    dataloadermanager.bas_addr_cur = 0
    datastorermanager.base_addrs_res_cur = 0
    for cal_idx in range(cal_total_num):
        for load_per_cal_index in range(load_num_per_cal):
            # DataLoader
            if cal_idx < padding_num:
                is_padding_row = 1
            elif cal_idx > cal_total_num - 1 - padding_num:
                is_padding_row = 5
            else:
                is_padding_row = 0
            DataLoader.dispatch(
                layer_idx = layer_idx,
                line_buffer_reshape = 0,
                is_padding_row = is_padding_row,
                read_mode = 0,
                transnum = 4,
                line_buffer_idx = dataloadermanager.line_buffer_idx,
                src_buffer_idx = 'a',
                bas_addr = dataloadermanager.bas_addr_cur + 144 * load_per_cal_index
            )
            dataloadermanager.line_buffer_idx_switch()
           
            # WeightLoader
            WeightLoader.dispatch(
                acc_reg_comp_idx = weightloadermanager.acc_reg_idx,
                kernal_size = 0,
                line_buffer_row_shift = 1,
                line_buffer_idx = weightloadermanager.line_buffer_idx,
                is_padding_col = 1,
                weight_parall_mode = 0,
                is_new = 0 if load_per_cal_index == 0 else 1, # 开始新的时候覆盖
                transnum = 9,
                bas_addr = weightloadermanager.bas_addr_cur[0] + load_per_cal_index * 9, 
                is_bilinear_bicubic = 0,
                offset_reg_idx = 0
            )
            weightloadermanager.line_buffer_idx_switch()

        # DataStorer
        DataStorer.dispatch(
            quant_config_idx = datastorermanager.quant_config_idx,
            pixelshuffle_out_mode = 0,
            is_pixelshuffle = 0,
            pooling_out_mode = 0,
            is_pooling = 0,
            reg_out_idx = datastorermanager.acc_reg_idx,
            acc_mode = 0,
            transfer_num = 1,
            store_mode = 0,
            stride = 144,
            base_addr_pooling = 0,
            base_addrs_res = datastorermanager.base_addrs_res_cur,
            is_bicubic_add = 0,
            is_first_or_last_row = 0,
            is_new = 0,
            dest_buffer_idx = 'b' # 存到b
        )
        datastorermanager.acc_reg_idx_switch()
        datastorermanager.base_addrs_res_cur += 2

        weightloadermanager.acc_reg_idx_switch()
        dataloadermanager.bas_addr_cur += 1 if cal_idx < padding_num else 2

    # 右半部分128*144
    # 初始变量设定
    cal_total_num  = 144//2 # 一次算出2行
    load_num_per_cal = 4 # 遍历c_in
    load_total_num = cal_total_num * load_num_per_cal
    dataloadermanager.line_buffer_load_total_num = load_total_num
    padding_num = 1
    dataloadermanager.bas_addr_cur = 144*4
    datastorermanager.base_addrs_res_cur = 144*4
    for cal_idx in range(cal_total_num):
        for load_per_cal_index in range(load_num_per_cal):
            # DataLoader
            if cal_idx < padding_num:
                is_padding_row = 1
            elif cal_idx > cal_total_num - 1 - padding_num:
                is_padding_row = 5
            else:
                is_padding_row = 0
            DataLoader.dispatch(
                layer_idx = layer_idx,
                line_buffer_reshape = 0,
                is_padding_row = is_padding_row,
                read_mode = 0,
                transnum = 4,
                line_buffer_idx = dataloadermanager.line_buffer_idx,
                src_buffer_idx = 'a',
                bas_addr = dataloadermanager.bas_addr_cur + 144 * load_per_cal_index
            )
            dataloadermanager.line_buffer_idx_switch()
           
            # WeightLoader
            WeightLoader.dispatch(
                acc_reg_comp_idx = weightloadermanager.acc_reg_idx,
                kernal_size = 0,
                line_buffer_row_shift = 1,
                line_buffer_idx = weightloadermanager.line_buffer_idx,
                is_padding_col = 1,
                weight_parall_mode = 0,
                is_new = 0 if load_per_cal_index == 0 else 1, # 开始新的时候覆盖
                transnum = 9,
                bas_addr = weightloadermanager.bas_addr_cur[0] + load_per_cal_index * 9, 
                is_bilinear_bicubic = 0,
                offset_reg_idx = 0
            )
            weightloadermanager.line_buffer_idx_switch()

        # DataStorer
        DataStorer.dispatch(
            quant_config_idx = datastorermanager.quant_config_idx,
            pixelshuffle_out_mode = 0,
            is_pixelshuffle = 0,
            pooling_out_mode = 0,
            is_pooling = 0,
            reg_out_idx = datastorermanager.acc_reg_idx,
            acc_mode = 0,
            transfer_num = 1,
            store_mode = 0,
            stride = 144,
            base_addr_pooling = 0,
            base_addrs_res = datastorermanager.base_addrs_res_cur,
            is_bicubic_add = 0,
            is_first_or_last_row = 0,
            is_new = 0,
            dest_buffer_idx = 'b' # 存到b
        )
        datastorermanager.acc_reg_idx_switch()
        datastorermanager.base_addrs_res_cur += 2

        weightloadermanager.acc_reg_idx_switch()
        dataloadermanager.bas_addr_cur += 1 if cal_idx < padding_num else 2

    # 收尾
    weightloadermanager.bas_addr_cur[0] += 4*9 
    datastorermanager.buffer_b_model.append({"begin":0, "end":144*4*2, "info":"c1_layer1_out", "valid":True})
    datastorermanager.buffer_a_model[0]["valid"] = False
    quantloadermanager.quant_config_idx_switch()
    datastorermanager.quant_config_idx_switch()

    # layer 2  256*144 conv1_2
    layer_idx = 2
    quant_mode = 0
    QuantLoader.dispatch(
        quant_reg_load_idx = quantloadermanager.quant_config_idx,
        quant_mode = quant_mode,
        layer_idx = layer_idx,
        transnum = 4,
        bas_addr = quantloadermanager.bas_addr_cur
    )
    quantloadermanager.bas_addr_cur += 4

    # 左半部分128*144
    # 初始变量设定
    cal_total_num  = 144//2 # 一次算出2行
    load_num_per_cal = 4 # 遍历c_in
    load_total_num = cal_total_num * load_num_per_cal
    dataloadermanager.line_buffer_load_total_num = load_total_num
    padding_num = 1
    dataloadermanager.bas_addr_cur = 0
    datastorermanager.base_addrs_res_cur = 0
    datastorermanager.base_addr_pooling_cur = 144*4*2 # 此层需要pooling，要预留好待cat的全部地址空间
    for cal_idx in range(cal_total_num):
        for load_per_cal_index in range(load_num_per_cal):
            # DataLoader
            if cal_idx < padding_num:
                is_padding_row = 1
            elif cal_idx > cal_total_num - 1 - padding_num:
                is_padding_row = 5
            else:
                is_padding_row = 0
            DataLoader.dispatch(
                layer_idx = layer_idx,
                line_buffer_reshape = 0,
                is_padding_row = is_padding_row,
                read_mode = 0,
                transnum = 4,
                line_buffer_idx = dataloadermanager.line_buffer_idx,
                src_buffer_idx = 'b',
                bas_addr = dataloadermanager.bas_addr_cur + 144 * load_per_cal_index
            )
            dataloadermanager.line_buffer_idx_switch()
           
            # WeightLoader
            WeightLoader.dispatch(
                acc_reg_comp_idx = weightloadermanager.acc_reg_idx,
                kernal_size = 0,
                line_buffer_row_shift = 1,
                line_buffer_idx = weightloadermanager.line_buffer_idx,
                is_padding_col = 1,
                weight_parall_mode = 0,
                is_new = 0 if load_per_cal_index == 0 else 1, # 开始新的时候覆盖
                transnum = 9,
                bas_addr = weightloadermanager.bas_addr_cur[0] + load_per_cal_index * 9 , 
                is_bilinear_bicubic = 0,
                offset_reg_idx = 0
            )
            weightloadermanager.line_buffer_idx_switch()

        # DataStorer
        DataStorer.dispatch(
            quant_config_idx = datastorermanager.quant_config_idx,
            pixelshuffle_out_mode = 0,
            is_pixelshuffle = 0,
            pooling_out_mode = 0,
            is_pooling = 1,
            reg_out_idx = datastorermanager.acc_reg_idx,
            acc_mode = 0,
            transfer_num = 1,
            store_mode = 0,
            stride = 144,
            base_addr_pooling = datastorermanager.base_addr_pooling_cur,
            base_addrs_res = datastorermanager.base_addrs_res_cur,
            is_bicubic_add = 0,
            is_first_or_last_row = 0,
            is_new = 0,
            dest_buffer_idx = 'a' # 存到a
        )
        datastorermanager.acc_reg_idx_switch()
        datastorermanager.base_addrs_res_cur += 2
        datastorermanager.base_addr_pooling_cur += 4

        weightloadermanager.acc_reg_idx_switch()
        dataloadermanager.bas_addr_cur += 1 if cal_idx < padding_num else 2

    # 右半部分128*144
    # 初始变量设定
    cal_total_num  = 144//2 # 一次算出2行
    load_num_per_cal = 4 # 遍历c_in
    load_total_num = cal_total_num * load_num_per_cal
    dataloadermanager.line_buffer_load_total_num = load_total_num
    padding_num = 1
    dataloadermanager.bas_addr_cur = 144*4
    datastorermanager.base_addrs_res_cur = 144*4
    datastorermanager.base_addr_pooling_cur = 144*4*2 + 2 # 此层需要pooling，要预留好待cat的全部地址空间
    for cal_idx in range(cal_total_num):
        for load_per_cal_index in range(load_num_per_cal):
            # DataLoader
            if cal_idx < padding_num:
                is_padding_row = 1
            elif cal_idx > cal_total_num - 1 - padding_num:
                is_padding_row = 5
            else:
                is_padding_row = 0
            DataLoader.dispatch(
                layer_idx = layer_idx,
                line_buffer_reshape = 0,
                is_padding_row = is_padding_row,
                read_mode = 0,
                transnum = 4,
                line_buffer_idx = dataloadermanager.line_buffer_idx,
                src_buffer_idx = 'b',
                bas_addr = dataloadermanager.bas_addr_cur + 144 * load_per_cal_index
            )
            dataloadermanager.line_buffer_idx_switch()
           
            # WeightLoader
            WeightLoader.dispatch(
                acc_reg_comp_idx = weightloadermanager.acc_reg_idx,
                kernal_size = 0,
                line_buffer_row_shift = 1,
                line_buffer_idx = weightloadermanager.line_buffer_idx,
                is_padding_col = 1,
                weight_parall_mode = 0,
                is_new = 0 if load_per_cal_index == 0 else 1, # 开始新的时候覆盖
                transnum = 9,
                bas_addr = weightloadermanager.bas_addr_cur[0] + load_per_cal_index * 9, 
                is_bilinear_bicubic = 0,
                offset_reg_idx = 0
            )
            weightloadermanager.line_buffer_idx_switch()

        # DataStorer
        DataStorer.dispatch(
            quant_config_idx = datastorermanager.quant_config_idx,
            pixelshuffle_out_mode = 0,
            is_pixelshuffle = 0,
            pooling_out_mode = 0,
            is_pooling = 1,
            reg_out_idx = datastorermanager.acc_reg_idx,
            acc_mode = 0,
            transfer_num = 1,
            store_mode = 0,
            stride = 144,
            base_addr_pooling = datastorermanager.base_addr_pooling_cur,
            base_addrs_res = datastorermanager.base_addrs_res_cur,
            is_bicubic_add = 0,
            is_first_or_last_row = 0,
            is_new = 0,
            dest_buffer_idx = 'a' # 存到a
        )
        datastorermanager.acc_reg_idx_switch()
        datastorermanager.base_addrs_res_cur += 2
        datastorermanager.base_addr_pooling_cur += 4

        weightloadermanager.acc_reg_idx_switch()
        dataloadermanager.bas_addr_cur += 1 if cal_idx < padding_num else 2

    # 收尾
    weightloadermanager.bas_addr_cur[0] += 4*9  
    datastorermanager.buffer_a_model.append({"begin":0, "end":144*4*2, "info":"c1_for_cat", "valid":True})
    datastorermanager.buffer_a_model.append({"begin":144*4*2, "end":144*4*2+72*4, "info":"c1_pool_out", "valid":True})
    datastorermanager.buffer_b_model[0]["valid"] = False
    quantloadermanager.quant_config_idx_switch()
    datastorermanager.quant_config_idx_switch()

    # layer 3  128*72 conv2
    layer_idx = 3
    quant_mode = 1
    QuantLoader.dispatch(
        quant_reg_load_idx = quantloadermanager.quant_config_idx,
        quant_mode = quant_mode,
        layer_idx = layer_idx,
        transnum = 8,
        bas_addr = quantloadermanager.bas_addr_cur
    )
    quantloadermanager.bas_addr_cur += 8

    # 初始变量设定
    cal_total_num  = 72 # 一次算出1行
    load_num_per_cal = 3 # 遍历ky
    load_total_num = cal_total_num * load_num_per_cal
    dataloadermanager.line_buffer_load_total_num = load_total_num
    padding_num = 1
    dataloadermanager.bas_addr_cur = 144*4*2 # 已经存了c1
    datastorermanager.base_addrs_res_cur = 0
    datastorermanager.base_addr_pooling_cur = 0
    for cal_idx in range(cal_total_num):
        for load_per_cal_index in range(load_num_per_cal):
            # DataLoader
            if cal_idx < padding_num and load_per_cal_index < padding_num:
                is_padding_row = 2
            elif cal_idx > cal_total_num - 1 - padding_num and load_per_cal_index > load_num_per_cal - 1 - padding_num:
                is_padding_row = 2
            else:
                is_padding_row = 0
            DataLoader.dispatch(
                layer_idx = layer_idx,
                line_buffer_reshape = 1,
                is_padding_row = is_padding_row,
                read_mode = 0,
                transnum = 4,
                line_buffer_idx = dataloadermanager.line_buffer_idx,
                src_buffer_idx = 'a',
                bas_addr = dataloadermanager.bas_addr_cur + 4 * load_per_cal_index - (4 if cal_idx < padding_num and (load_per_cal_index == 1 or load_per_cal_index == 2) else 0)
            )
            dataloadermanager.line_buffer_idx_switch()
           
            # WeightLoader
            WeightLoader.dispatch(
                acc_reg_comp_idx = weightloadermanager.acc_reg_idx,
                kernal_size = 0,
                line_buffer_row_shift = 0,
                line_buffer_idx = weightloadermanager.line_buffer_idx,
                is_padding_col = 1,
                weight_parall_mode = 0,
                is_new = 0 if load_per_cal_index == 0 else 1, # 开始新的时候覆盖
                transnum = 12,
                bas_addr = weightloadermanager.bas_addr_cur[0] + 12 * load_per_cal_index, # 需要递增
                is_bilinear_bicubic = 0,
                offset_reg_idx = 0
            )
            weightloadermanager.line_buffer_idx_switch()

        # DataStorer
        DataStorer.dispatch(
            quant_config_idx = datastorermanager.quant_config_idx,
            pixelshuffle_out_mode = 0,
            is_pixelshuffle = 0,
            pooling_out_mode = 0,
            is_pooling = 0,
            reg_out_idx = datastorermanager.acc_reg_idx,
            acc_mode = 1,
            transfer_num = 1,
            store_mode = 1,
            stride = 0,
            base_addr_pooling = datastorermanager.base_addr_pooling_cur,
            base_addrs_res = datastorermanager.base_addrs_res_cur,
            is_bicubic_add = 0,
            is_first_or_last_row = 0,
            is_new = 0,
            dest_buffer_idx = 'b' # 存到b
        )
        datastorermanager.acc_reg_idx_switch()
        datastorermanager.base_addrs_res_cur += 8

        weightloadermanager.acc_reg_idx_switch()
        dataloadermanager.bas_addr_cur += 0 if cal_idx < padding_num else 4

    # 收尾
    weightloadermanager.bas_addr_cur[0] += 3*12 
    datastorermanager.buffer_b_model.append({"begin":0, "end":72*8, "info":"c2_out", "valid":True})
    datastorermanager.buffer_a_model[-1]["valid"] = False
    quantloadermanager.quant_config_idx_switch()
    datastorermanager.quant_config_idx_switch()

    # layer 4  128*72 conv3
    layer_idx = 4
    quant_mode = 1
    QuantLoader.dispatch(
        quant_reg_load_idx = quantloadermanager.quant_config_idx,
        quant_mode = quant_mode,
        layer_idx = layer_idx,
        transnum = 8,
        bas_addr = quantloadermanager.bas_addr_cur
    )
    quantloadermanager.bas_addr_cur += 8

    # 初始变量设定
    cal_total_num  = 72 # 一次算出1行
    ic_load_num_per_cal = 2 # 遍历ic
    ky_load_num_per_cal = 3 # 遍历ky
    load_total_num = cal_total_num * ic_load_num_per_cal * ky_load_num_per_cal
    dataloadermanager.line_buffer_load_total_num = load_total_num
    padding_num = 1
    dataloadermanager.bas_addr_cur = 0
    datastorermanager.base_addrs_res_cur = 144*4*2 # 已经存了c1
    datastorermanager.base_addr_pooling_cur = 144*4*2 + 72*8 # c1、c3
    for cal_idx in range(cal_total_num):
        for ky_load_num_per_cal_index in range(ky_load_num_per_cal):
            for ic_load_num_per_cal_index in range(ic_load_num_per_cal):
                # DataLoader
                if cal_idx < padding_num and ky_load_num_per_cal_index < padding_num:
                    is_padding_row = 2
                elif cal_idx > cal_total_num - 1 - padding_num and ky_load_num_per_cal_index > ky_load_num_per_cal - 1 - padding_num:
                    is_padding_row = 2
                else:
                    is_padding_row = 0
                DataLoader.dispatch(
                    layer_idx = layer_idx,
                    line_buffer_reshape = 0,
                    is_padding_row = is_padding_row,
                    read_mode = 0,
                    transnum = 4,
                    line_buffer_idx = dataloadermanager.line_buffer_idx,
                    src_buffer_idx = 'b',
                    bas_addr = dataloadermanager.bas_addr_cur + 8 * ky_load_num_per_cal_index + 4 * ic_load_num_per_cal_index - (8 if cal_idx < padding_num and (ky_load_num_per_cal_index == 1 or ky_load_num_per_cal_index == 2) else 0)
                )
                dataloadermanager.line_buffer_idx_switch()
            
                # WeightLoader
                WeightLoader.dispatch(
                    acc_reg_comp_idx = weightloadermanager.acc_reg_idx,
                    kernal_size = 0,
                    line_buffer_row_shift = 0,
                    line_buffer_idx = weightloadermanager.line_buffer_idx,
                    is_padding_col = 1,
                    weight_parall_mode = 0,
                    is_new = 0 if ky_load_num_per_cal_index == 0 and ic_load_num_per_cal_index == 0 else 1, # 开始新的时候覆盖
                    transnum = 12,
                    bas_addr = weightloadermanager.bas_addr_cur[0] + 12 * (ky_load_num_per_cal_index * ic_load_num_per_cal + ic_load_num_per_cal_index), # 需要递增
                    is_bilinear_bicubic = 0,
                    offset_reg_idx = 0
                )
                weightloadermanager.line_buffer_idx_switch()

        # DataStorer
        DataStorer.dispatch(
            quant_config_idx = datastorermanager.quant_config_idx,
            pixelshuffle_out_mode = 0,
            is_pixelshuffle = 0,
            pooling_out_mode = 1,
            is_pooling = 1,
            reg_out_idx = datastorermanager.acc_reg_idx,
            acc_mode = 1,
            transfer_num = 1,
            store_mode = 1,
            stride = 0,
            base_addr_pooling = datastorermanager.base_addr_pooling_cur,
            base_addrs_res = datastorermanager.base_addrs_res_cur,
            is_bicubic_add = 0,
            is_first_or_last_row = 0,
            is_new = 0,
            dest_buffer_idx = 'a' # 存到a
        )
        datastorermanager.acc_reg_idx_switch()
        datastorermanager.base_addrs_res_cur += 8
        if cal_idx % 2 == 1: # 每两次计算才会得出一次池化结果
            datastorermanager.base_addr_pooling_cur += 4

        weightloadermanager.acc_reg_idx_switch()
        dataloadermanager.bas_addr_cur += 0 if cal_idx < padding_num else 8

    # 收尾
    weightloadermanager.bas_addr_cur[0] += 6*12 
    datastorermanager.buffer_a_model.append({"begin":144*4*2, "end":144*4*2 + 72*8, "info":"c3_for_cat", "valid":True})
    datastorermanager.buffer_a_model.append({"begin":144*4*2 + 72*8, "end":144*4*2 + 72*8 + 36*4, "info":"c3_pool_out", "valid":True})
    datastorermanager.buffer_b_model[-1]["valid"] = False
    quantloadermanager.quant_config_idx_switch()
    datastorermanager.quant_config_idx_switch()
    
    # layer 5  64*36 conv4
    layer_idx = 5
    quant_mode = 2
    QuantLoader.dispatch(
        quant_reg_load_idx = quantloadermanager.quant_config_idx,
        quant_mode = quant_mode,
        layer_idx = layer_idx,
        transnum = 16,
        bas_addr = quantloadermanager.bas_addr_cur
    )
    quantloadermanager.bas_addr_cur += 16

    # 初始变量设定
    cal_total_num = 36 # 一次算出1行
    ky_load_num_per_cal = 3 # 遍历ky
    load_total_num = cal_total_num * ky_load_num_per_cal
    dataloadermanager.line_buffer_load_total_num = load_total_num
    padding_num = 1
    dataloadermanager.bas_addr_cur = 144*4*2 + 72*8 # 已经存了c1、c3
    datastorermanager.base_addrs_res_cur = 0
    datastorermanager.base_addr_pooling_cur = 0
    for cal_idx in range(cal_total_num):
        for ky_load_num_per_cal_index in range(ky_load_num_per_cal):
            # DataLoader
            if cal_idx < padding_num and ky_load_num_per_cal_index < padding_num:
                is_padding_row = 2
            elif cal_idx > cal_total_num - 1 - padding_num and ky_load_num_per_cal_index > ky_load_num_per_cal - 1 - padding_num:
                is_padding_row = 2
            else:
                is_padding_row = 0
            DataLoader.dispatch(
                layer_idx = layer_idx,
                line_buffer_reshape = 0,
                is_padding_row = is_padding_row,
                read_mode = 0,
                transnum = 4,
                line_buffer_idx = dataloadermanager.line_buffer_idx,
                src_buffer_idx = 'a',
                bas_addr = dataloadermanager.bas_addr_cur + 4 * ky_load_num_per_cal_index - (4 if cal_idx < padding_num and (ky_load_num_per_cal_index == 1 or ky_load_num_per_cal_index == 2) else 0)
            )
            dataloadermanager.line_buffer_idx_switch()
        
            # WeightLoader
            WeightLoader.dispatch(
                acc_reg_comp_idx = weightloadermanager.acc_reg_idx,
                kernal_size = 0,
                line_buffer_row_shift = 3,
                line_buffer_idx = weightloadermanager.line_buffer_idx,
                is_padding_col = 2,
                weight_parall_mode = 1,
                is_new = 0 if ky_load_num_per_cal_index == 0 else 1, # 开始新的时候覆盖
                transnum = 24,
                bas_addr = weightloadermanager.bas_addr_cur[1] + 24 * ky_load_num_per_cal_index, # 需要递增
                is_bilinear_bicubic = 0,
                offset_reg_idx = 0
            )
            weightloadermanager.line_buffer_idx_switch()

        # DataStorer
        DataStorer.dispatch(
            quant_config_idx = datastorermanager.quant_config_idx,
            pixelshuffle_out_mode = 0,
            is_pixelshuffle = 0,
            pooling_out_mode = 0,
            is_pooling = 0,
            reg_out_idx = datastorermanager.acc_reg_idx,
            acc_mode = 1,
            transfer_num = 1,
            store_mode = 1,
            stride = 0,
            base_addr_pooling = datastorermanager.base_addr_pooling_cur,
            base_addrs_res = datastorermanager.base_addrs_res_cur,
            is_bicubic_add = 0,
            is_first_or_last_row = 0,
            is_new = 0,
            dest_buffer_idx = 'b' # 存到b
        )
        datastorermanager.acc_reg_idx_switch()
        datastorermanager.base_addrs_res_cur += 8

        weightloadermanager.acc_reg_idx_switch()
        dataloadermanager.bas_addr_cur += 0 if cal_idx < padding_num else 4

    # 收尾
    weightloadermanager.bas_addr_cur[1] += 3*24 
    datastorermanager.buffer_b_model.append({"begin":0, "end":36*8, "info":"c4_out", "valid":True})
    datastorermanager.buffer_a_model[-1]["valid"] = False
    quantloadermanager.quant_config_idx_switch()
    datastorermanager.quant_config_idx_switch()

    # layer 6  64*36 conv5
    layer_idx = 6
    quant_mode = 2
    QuantLoader.dispatch(
        quant_reg_load_idx = quantloadermanager.quant_config_idx,
        quant_mode = quant_mode,
        layer_idx = layer_idx,
        transnum = 16,
        bas_addr = quantloadermanager.bas_addr_cur
    )
    quantloadermanager.bas_addr_cur += 16

    # 初始变量设定
    cal_total_num  = 36 # 一次算出1行
    ic_load_num_per_cal = 2 # 遍历ic
    ky_load_num_per_cal = 3 # 遍历ky
    load_total_num = cal_total_num * ic_load_num_per_cal * ky_load_num_per_cal
    dataloadermanager.line_buffer_load_total_num = load_total_num
    padding_num = 1
    dataloadermanager.bas_addr_cur = 0
    datastorermanager.base_addrs_res_cur = 144*4*2 + 72*8 # 已经存了c1、c3
    datastorermanager.base_addr_pooling_cur = 144*4*2 + 72*8 + 36*8 # c1、c3、c5
    for cal_idx in range(cal_total_num):
        for ky_load_num_per_cal_index in range(ky_load_num_per_cal):
            for ic_load_num_per_cal_index in range(ic_load_num_per_cal):
                # DataLoader
                if cal_idx < padding_num and ky_load_num_per_cal_index < padding_num:
                    is_padding_row = 2
                elif cal_idx > cal_total_num - 1 - padding_num and ky_load_num_per_cal_index > ky_load_num_per_cal - 1 - padding_num:
                    is_padding_row = 2
                else:
                    is_padding_row = 0
                DataLoader.dispatch(
                    layer_idx = layer_idx,
                    line_buffer_reshape = 0,
                    is_padding_row = is_padding_row,
                    read_mode = 0,
                    transnum = 4,
                    line_buffer_idx = dataloadermanager.line_buffer_idx,
                    src_buffer_idx = 'b',
                    bas_addr = dataloadermanager.bas_addr_cur + 8 * ky_load_num_per_cal_index + 4 * ic_load_num_per_cal_index - (8 if cal_idx < padding_num and (ky_load_num_per_cal_index == 1 or ky_load_num_per_cal_index == 2) else 0)
                )
                dataloadermanager.line_buffer_idx_switch()
            
                # WeightLoader
                WeightLoader.dispatch(
                    acc_reg_comp_idx = weightloadermanager.acc_reg_idx,
                    kernal_size = 0,
                    line_buffer_row_shift = 3,
                    line_buffer_idx = weightloadermanager.line_buffer_idx,
                    is_padding_col = 2,
                    weight_parall_mode = 1,
                    is_new = 0 if ky_load_num_per_cal_index == 0 and ic_load_num_per_cal_index == 0 else 1, # 开始新的时候覆盖
                    transnum = 24,
                    bas_addr = weightloadermanager.bas_addr_cur[1] + 24 * (ky_load_num_per_cal_index * ic_load_num_per_cal + ic_load_num_per_cal_index), # 需要递增
                    is_bilinear_bicubic = 0,
                    offset_reg_idx = 0
                )
                weightloadermanager.line_buffer_idx_switch()

        # DataStorer
        DataStorer.dispatch(
            quant_config_idx = datastorermanager.quant_config_idx,
            pixelshuffle_out_mode = 0,
            is_pixelshuffle = 0,
            pooling_out_mode = 1,
            is_pooling = 1,
            reg_out_idx = datastorermanager.acc_reg_idx,
            acc_mode = 1,
            transfer_num = 1,
            store_mode = 1,
            stride = 0,
            base_addr_pooling = datastorermanager.base_addr_pooling_cur,
            base_addrs_res = datastorermanager.base_addrs_res_cur,
            is_bicubic_add = 0,
            is_first_or_last_row = 0,
            is_new = 0,
            dest_buffer_idx = 'a' # 存到a
        )
        datastorermanager.acc_reg_idx_switch()
        datastorermanager.base_addrs_res_cur += 8
        if cal_idx % 2 == 1:
            datastorermanager.base_addr_pooling_cur += 4

        weightloadermanager.acc_reg_idx_switch()
        dataloadermanager.bas_addr_cur += 0 if cal_idx < padding_num else 8

    # 收尾
    weightloadermanager.bas_addr_cur[1] += 6*24 
    datastorermanager.buffer_a_model.append({"begin":144*4*2 + 72*8, "end":144*4*2 + 72*8 + 36*8, "info":"c5_for_cat", "valid":True})
    datastorermanager.buffer_a_model.append({"begin":144*4*2 + 72*8 + 36*8, "end":144*4*2 + 72*8 + 36*8 + 18*4, "info":"c5_pool_out", "valid":True})
    datastorermanager.buffer_b_model[-1]["valid"] = False
    quantloadermanager.quant_config_idx_switch()
    datastorermanager.quant_config_idx_switch()
    
    # layer 7  32*18 conv6
    layer_idx = 7
    group = 2
    for group_idx in range(group):
        quant_mode = 3
        QuantLoader.dispatch(
            quant_reg_load_idx = quantloadermanager.quant_config_idx,
            quant_mode = quant_mode,
            layer_idx = layer_idx,
            transnum = 32,
            bas_addr = quantloadermanager.bas_addr_cur
        )
        quantloadermanager.bas_addr_cur += 32

        # 初始变量设定
        cal_total_num = 18 # 一次算出1行
        ky_load_num_per_cal = 3 # 遍历ky
        load_total_num = cal_total_num * ky_load_num_per_cal
        dataloadermanager.line_buffer_load_total_num = load_total_num
        padding_num = 1
        dataloadermanager.bas_addr_cur = 144*4*2 + 72*8 + 36*8 + group_idx * 2 # 已经存了c1、c3、c5
        datastorermanager.base_addrs_res_cur = 0 + group_idx * 18 * 8
        datastorermanager.base_addr_pooling_cur = 0
        for cal_idx in range(cal_total_num):
            for ky_load_num_per_cal_index in range(ky_load_num_per_cal):
                # DataLoader
                if cal_idx < padding_num and ky_load_num_per_cal_index < padding_num:
                    is_padding_row = 2
                elif cal_idx > cal_total_num - 1 - padding_num and ky_load_num_per_cal_index > ky_load_num_per_cal - 1 - padding_num:
                    is_padding_row = 2
                else:
                    is_padding_row = 0
                DataLoader.dispatch(
                    layer_idx = layer_idx,
                    line_buffer_reshape = 0,
                    is_padding_row = is_padding_row,
                    read_mode = 0,
                    transnum = 2,
                    line_buffer_idx = dataloadermanager.line_buffer_idx,
                    src_buffer_idx = 'a',
                    bas_addr = dataloadermanager.bas_addr_cur + 4 * ky_load_num_per_cal_index - (4 if cal_idx < padding_num and (ky_load_num_per_cal_index == 1 or ky_load_num_per_cal_index == 2) else 0)
                )
                dataloadermanager.line_buffer_idx_switch()
            
                # WeightLoader
                WeightLoader.dispatch(
                    acc_reg_comp_idx = weightloadermanager.acc_reg_idx,
                    kernal_size = 0,
                    line_buffer_row_shift = 4,
                    line_buffer_idx = weightloadermanager.line_buffer_idx,
                    is_padding_col = 3,
                    weight_parall_mode = 2,
                    is_new = 0 if ky_load_num_per_cal_index == 0 else 1, # 开始新的时候覆盖
                    transnum = 24,
                    bas_addr = weightloadermanager.bas_addr_cur[2] + 24 * ky_load_num_per_cal_index, # 需要递增
                    is_bilinear_bicubic = 0,
                    offset_reg_idx = 0
                )
                weightloadermanager.line_buffer_idx_switch()

            # DataStorer
            DataStorer.dispatch(
                quant_config_idx = datastorermanager.quant_config_idx,
                pixelshuffle_out_mode = 0,
                is_pixelshuffle = 0,
                pooling_out_mode = 0,
                is_pooling = 0,
                reg_out_idx = datastorermanager.acc_reg_idx,
                acc_mode = 1,
                transfer_num = 1,
                store_mode = 1,
                stride = 0,
                base_addr_pooling = datastorermanager.base_addr_pooling_cur,
                base_addrs_res = datastorermanager.base_addrs_res_cur,
                is_bicubic_add = 0,
                is_first_or_last_row = 0,
                is_new = 0,
                dest_buffer_idx = 'b' # 存到b
            )
            datastorermanager.acc_reg_idx_switch()
            datastorermanager.base_addrs_res_cur += 8

            weightloadermanager.acc_reg_idx_switch()
            dataloadermanager.bas_addr_cur += 0 if cal_idx < padding_num else 4

        # 收尾
        weightloadermanager.bas_addr_cur[2] += 3*24
        if group_idx == 1:
            datastorermanager.buffer_b_model.append({"begin":0, "end":18*8*2, "info":f"c6_out", "valid":True})
            datastorermanager.buffer_a_model[-1]["valid"] = False
        quantloadermanager.quant_config_idx_switch()
        datastorermanager.quant_config_idx_switch()

    # layer 8  32*18 conv7 (64,64,3,8)
    layer_idx = 8
    # 8个group，分为2组4个算
    group_level1 = 2
    group_level2 = 4
    for group_level1_idx in range(group_level1):
        quant_mode = 3
        QuantLoader.dispatch(
            quant_reg_load_idx = quantloadermanager.quant_config_idx,
            quant_mode = quant_mode,
            layer_idx = layer_idx,
            transnum = 32,
            bas_addr = quantloadermanager.bas_addr_cur
        )
        quantloadermanager.bas_addr_cur += 32

        # 初始变量设定
        cal_total_num  = 18 # 一次算出1行
        ic_load_num_per_cal = 2 # 遍历ic
        ky_load_num_per_cal = 3 # 遍历ky
        load_total_num = cal_total_num * ic_load_num_per_cal * ky_load_num_per_cal
        dataloadermanager.line_buffer_load_total_num = load_total_num
        padding_num = 1
        dataloadermanager.bas_addr_cur = 0 + group_level1_idx*18*8
        datastorermanager.base_addrs_res_cur = 144*4*2 + 72*8 + 36*8 + 9*8*2 + group_level1_idx*18*8 # 已经存了c1、c3、c5 和 c7_pool_out
        datastorermanager.base_addr_pooling_cur = 144*4*2 + 72*8 + 36*8 + group_level1_idx*9*8 # c1、c3、c5
        for cal_idx in range(cal_total_num):
            for ky_load_num_per_cal_index in range(ky_load_num_per_cal):
                for ic_load_num_per_cal_index in range(ic_load_num_per_cal):
                    # DataLoader
                    if cal_idx < padding_num and ky_load_num_per_cal_index < padding_num:
                        is_padding_row = 2
                    elif cal_idx > cal_total_num - 1 - padding_num and ky_load_num_per_cal_index > ky_load_num_per_cal - 1 - padding_num:
                        is_padding_row = 2
                    else:
                        is_padding_row = 0
                    DataLoader.dispatch(
                        layer_idx = layer_idx,
                        line_buffer_reshape = 0,
                        is_padding_row = is_padding_row,
                        read_mode = 0,
                        transnum = 4,
                        line_buffer_idx = dataloadermanager.line_buffer_idx,
                        src_buffer_idx = 'b',
                        bas_addr = dataloadermanager.bas_addr_cur + 8 * ky_load_num_per_cal_index + 4 * ic_load_num_per_cal_index - (8 if cal_idx < padding_num and (ky_load_num_per_cal_index == 1 or ky_load_num_per_cal_index == 2) else 0)
                    )
                    dataloadermanager.line_buffer_idx_switch()
                
                    # WeightLoader
                    WeightLoader.dispatch(
                        acc_reg_comp_idx = weightloadermanager.acc_reg_idx,
                        kernal_size = 0,
                        line_buffer_row_shift = 0,
                        line_buffer_idx = weightloadermanager.line_buffer_idx,
                        is_padding_col = 3,
                        weight_parall_mode = 2,
                        is_new = 0 if ky_load_num_per_cal_index == 0 and ic_load_num_per_cal_index == 0 else 1, # 开始新的时候覆盖
                        transnum = 12,
                        bas_addr = weightloadermanager.bas_addr_cur[2] + 12 * (ky_load_num_per_cal_index * ic_load_num_per_cal + ic_load_num_per_cal_index), # 需要递增
                        is_bilinear_bicubic = 0,
                        offset_reg_idx = 0
                    )
                    weightloadermanager.line_buffer_idx_switch()

            # DataStorer
            DataStorer.dispatch(
                quant_config_idx = datastorermanager.quant_config_idx,
                pixelshuffle_out_mode = 0,
                is_pixelshuffle = 0,
                pooling_out_mode = 2,
                is_pooling = 1,
                reg_out_idx = datastorermanager.acc_reg_idx,
                acc_mode = 1,
                transfer_num = 1,
                store_mode = 2,
                stride = 18, # 此时是h连续
                base_addr_pooling = datastorermanager.base_addr_pooling_cur,
                base_addrs_res = datastorermanager.base_addrs_res_cur,
                is_bicubic_add = 0,
                is_first_or_last_row = 0,
                is_new = 0,
                dest_buffer_idx = 'a' # 存到a
            )
            datastorermanager.acc_reg_idx_switch()
            datastorermanager.base_addrs_res_cur += 1 # 此时是h连续
            if cal_idx % 2 == 1:
                datastorermanager.base_addr_pooling_cur += 8

            weightloadermanager.acc_reg_idx_switch()
            dataloadermanager.bas_addr_cur += 0 if cal_idx < padding_num else 8

        # 收尾
        weightloadermanager.bas_addr_cur[2] += 6*12 
        if group_level1_idx == 1: # 只append一次即可，因为这64个channel在后续卷积中不分组，没有必要带上group信息
            datastorermanager.buffer_a_model.append({"begin":144*4*2 + 72*8 + 36*8 + 9*8*2, "end":144*4*2 + 72*8 + 36*8 + 9*8*2 + 18*8*2, "info":"c7_for_cat", "valid":True})
            datastorermanager.buffer_a_model.append({"begin":144*4*2 + 72*8 + 36*8, "end":144*4*2 + 72*8 + 36*8 + 9*8*2, "info":"c7_pool_out", "valid":True})
        datastorermanager.buffer_b_model[-1]["valid"] = False
        quantloadermanager.quant_config_idx_switch()
        datastorermanager.quant_config_idx_switch()
    

    # layer 11  32*18 conv11 (128,16,3,2) 后半部分的(64,8,3) 注意后半部分带来的特殊地址
    layer_idx = 11
    for group_idx in [1]:
        quant_mode = 7
        QuantLoader.dispatch(
            quant_reg_load_idx = quantloadermanager.quant_config_idx,
            quant_mode = quant_mode,
            layer_idx = layer_idx,
            transnum = 8,
            bas_addr = quantloadermanager.bas_addr_cur + 64 + 256 + 8 # 手动越过了layer9,layer10,layer11的前半部分
        )
        # quantloadermanager.bas_addr_cur += 8

        # 初始变量设定
        cal_total_num = 18//4 + 1 # 一次算出4行 最后一次的transnum减半
        ic_load_num_per_cal = 16 # 遍历ic
        ky_load_num_per_cal = 3 # 遍历ky
        load_total_num = cal_total_num * ky_load_num_per_cal
        dataloadermanager.line_buffer_load_total_num = load_total_num
        padding_num = 1
        dataloadermanager.bas_addr_cur = 144*4*2 + 72*8 + 36*8 + 9*8*2 # 已经存了c1、c3、c5 和 c7_pool_out
        datastorermanager.base_addrs_res_cur = 0 
        datastorermanager.base_addr_pooling_cur = 0
        for cal_idx in range(cal_total_num):
            for ky_load_num_per_cal_index in range(ky_load_num_per_cal):
                for ic_load_num_per_cal_index in range(ic_load_num_per_cal):
                    # DataLoader
                    if cal_idx < padding_num and ky_load_num_per_cal_index < padding_num:
                        is_padding_row = 1
                    elif cal_idx > cal_total_num - 1 - padding_num and ky_load_num_per_cal_index > ky_load_num_per_cal - 1 - padding_num:
                        is_padding_row = 5
                    else:
                        is_padding_row = 0
                    DataLoader.dispatch(
                        layer_idx = layer_idx,
                        line_buffer_reshape = 3,
                        is_padding_row = is_padding_row,
                        read_mode = 0,
                        transnum = 4 if cal_idx < cal_total_num-1 else 2, 
                        line_buffer_idx = dataloadermanager.line_buffer_idx,
                        src_buffer_idx = 'a',
                        bas_addr = dataloadermanager.bas_addr_cur + 18 * ic_load_num_per_cal_index + ky_load_num_per_cal_index - (1 if cal_idx < padding_num and (ky_load_num_per_cal_index == 1 or ky_load_num_per_cal_index == 2) else 0) # 此时is_padding_row为0，从地址0处load出无padding的4行
                    )
                    dataloadermanager.line_buffer_idx_switch()
                
                    # WeightLoader
                    WeightLoader.dispatch(
                        acc_reg_comp_idx = weightloadermanager.acc_reg_idx,
                        kernal_size = 0,
                        line_buffer_row_shift = 0,
                        line_buffer_idx = weightloadermanager.line_buffer_idx,
                        is_padding_col = 3,
                        weight_parall_mode = 0,
                        is_new = 0 if ky_load_num_per_cal_index == 0 and ic_load_num_per_cal_index == 0 else 1, # 开始新的时候覆盖
                        transnum = 12,
                        bas_addr = weightloadermanager.bas_addr_cur[0] + 12 * (ky_load_num_per_cal_index*ic_load_num_per_cal + ic_load_num_per_cal_index), # 需要递增
                        is_bilinear_bicubic = 0,
                        offset_reg_idx = 0
                    )
                    weightloadermanager.line_buffer_idx_switch()

            # DataStorer
            DataStorer.dispatch(
                quant_config_idx = datastorermanager.quant_config_idx,
                pixelshuffle_out_mode = 0,
                is_pixelshuffle = 0,
                pooling_out_mode = 0,
                is_pooling = 0,
                reg_out_idx = datastorermanager.acc_reg_idx,
                acc_mode = 2,
                transfer_num = 1 if cal_idx < cal_total_num-1 else 0,
                store_mode = 1, 
                stride = 0,
                base_addr_pooling = datastorermanager.base_addr_pooling_cur,
                base_addrs_res = datastorermanager.base_addrs_res_cur,
                is_bicubic_add = 0,
                is_first_or_last_row = 0,
                is_new = 0,
                dest_buffer_idx = 'b' # 存到b
            )
            datastorermanager.acc_reg_idx_switch()
            datastorermanager.base_addrs_res_cur += 8

            weightloadermanager.acc_reg_idx_switch()
            dataloadermanager.bas_addr_cur += 3 if cal_idx < padding_num else 4

        # 收尾
        weightloadermanager.bas_addr_cur[0] += 12*48 
        datastorermanager.buffer_b_model.append({"begin":0 , "end":18*2, "info":f"c11_out_group{group_idx}", "valid":True})
        datastorermanager.buffer_a_model[-2]["valid"] = False
        quantloadermanager.quant_config_idx_switch()
        datastorermanager.quant_config_idx_switch()

    # layer 9  16*9 (64,64,3,8) conv8
    layer_idx = 9
    # 8个group，分为2组4个算
    group_level1 = 2
    group_level2 = 4
    for group_level1_idx in range(group_level1):
        quant_mode = 4
        QuantLoader.dispatch(
            quant_reg_load_idx = quantloadermanager.quant_config_idx,
            quant_mode = quant_mode,
            layer_idx = layer_idx,
            transnum = 32,
            bas_addr = quantloadermanager.bas_addr_cur
        )
        quantloadermanager.bas_addr_cur += 32

        # 初始变量设定
        cal_total_num = 9 # 一次算出1行
        ic_load_num_per_cal = 2 # 遍历ic
        ky_load_num_per_cal = 3 # 遍历ky
        load_total_num = cal_total_num * ic_load_num_per_cal * ky_load_num_per_cal
        dataloadermanager.line_buffer_load_total_num = load_total_num
        padding_num = 1
        dataloadermanager.bas_addr_cur = 144*4*2 + 72*8 + 36*8 + group_level1_idx*9*8
        datastorermanager.base_addrs_res_cur = 18*2 + group_level1_idx*9*4
        datastorermanager.base_addr_pooling_cur = 0
        for cal_idx in range(cal_total_num):
            for ky_load_num_per_cal_index in range(ky_load_num_per_cal):
                for ic_load_num_per_cal_index in range(ic_load_num_per_cal):
                    # DataLoader
                    if cal_idx < padding_num and ky_load_num_per_cal_index < padding_num:
                        is_padding_row = 2
                    elif cal_idx > cal_total_num - 1 - padding_num and ky_load_num_per_cal_index > ky_load_num_per_cal - 1 - padding_num:
                        is_padding_row = 2
                    else:
                        is_padding_row = 0
                    DataLoader.dispatch(
                        layer_idx = layer_idx,
                        line_buffer_reshape = 0,
                        is_padding_row = is_padding_row,
                        read_mode = 0,
                        transnum = 4,
                        line_buffer_idx = dataloadermanager.line_buffer_idx,
                        src_buffer_idx = 'a',
                        bas_addr = dataloadermanager.bas_addr_cur + 8 * ky_load_num_per_cal_index + 4 * ic_load_num_per_cal_index - (8 if cal_idx < padding_num and (ky_load_num_per_cal_index == 1 or ky_load_num_per_cal_index == 2) else 0)
                    )
                    dataloadermanager.line_buffer_idx_switch()
                
                    # WeightLoader
                    WeightLoader.dispatch(
                        acc_reg_comp_idx = weightloadermanager.acc_reg_idx,
                        kernal_size = 0,
                        line_buffer_row_shift = 0,
                        line_buffer_idx = weightloadermanager.line_buffer_idx,
                        is_padding_col = 4,
                        weight_parall_mode = 2,
                        is_new = 0 if ky_load_num_per_cal_index == 0 and ic_load_num_per_cal_index == 0 else 1, # 开始新的时候覆盖
                        transnum = 12,
                        bas_addr = weightloadermanager.bas_addr_cur[2] + 12 * (ky_load_num_per_cal_index * ic_load_num_per_cal + ic_load_num_per_cal_index), # 需要递增
                        is_bilinear_bicubic = 0,
                        offset_reg_idx = 0
                    )
                    weightloadermanager.line_buffer_idx_switch()

            # DataStorer
            DataStorer.dispatch(
                quant_config_idx = datastorermanager.quant_config_idx,
                pixelshuffle_out_mode = 0,
                is_pixelshuffle = 0,
                pooling_out_mode = 0,
                is_pooling = 0,
                reg_out_idx = datastorermanager.acc_reg_idx,
                acc_mode = 3,
                transfer_num = 0,
                store_mode = 1,
                stride = 0,
                base_addr_pooling = datastorermanager.base_addr_pooling_cur,
                base_addrs_res = datastorermanager.base_addrs_res_cur,
                is_bicubic_add = 0,
                is_first_or_last_row = 0,
                is_new = 0,
                dest_buffer_idx = 'b' # 存到b
            )
            datastorermanager.acc_reg_idx_switch()
            datastorermanager.base_addrs_res_cur += 4

            weightloadermanager.acc_reg_idx_switch()
            dataloadermanager.bas_addr_cur += 0 if cal_idx < padding_num else 8

        # 收尾
        weightloadermanager.bas_addr_cur[2] += 6*12 
        if group_level1_idx == 1: 
            datastorermanager.buffer_b_model.append({"begin":18*2, "end":18*2 + 9*4*2, "info":"c8_out", "valid":True})
        datastorermanager.buffer_a_model[-1]["valid"] = False
        quantloadermanager.quant_config_idx_switch()
        datastorermanager.quant_config_idx_switch()

    # layer 10  16*9 (64,256,3,8) conv10
    layer_idx = 10
    # 8个group，分为2组4个算
    group_level1 = 2
    group_level2 = 4
    for group_level1_idx in range(group_level1):
        for group_level2_idx in range(group_level2):
            quant_mode = 4
            QuantLoader.dispatch(
                quant_reg_load_idx = quantloadermanager.quant_config_idx,
                quant_mode = quant_mode,
                layer_idx = layer_idx,
                transnum = 32,
                bas_addr = quantloadermanager.bas_addr_cur
            )
            quantloadermanager.bas_addr_cur += 32

            # 初始变量设定
            cal_total_num  = 9 # 一次算出1行
            ky_load_num_per_cal = 3 # 遍历ky
            load_total_num = cal_total_num * ky_load_num_per_cal
            dataloadermanager.line_buffer_load_total_num = load_total_num
            padding_num = 1
            dataloadermanager.bas_addr_cur = 18*2 + group_level1_idx*9*4 + group_level2_idx
            datastorermanager.base_addrs_res_cur = 144*4*2 + 72*8 + 36*8 + group_level1_idx*18*8 + group_level2_idx * 18*2
            datastorermanager.base_addr_pooling_cur = 0
            for cal_idx in range(cal_total_num):
                for ky_load_num_per_cal_index in range(ky_load_num_per_cal):
                    # DataLoader
                    if cal_idx < padding_num and ky_load_num_per_cal_index < padding_num:
                        is_padding_row = 2
                    elif cal_idx > cal_total_num - 1 - padding_num and ky_load_num_per_cal_index > ky_load_num_per_cal - 1 - padding_num:
                        is_padding_row = 2
                    else:
                        is_padding_row = 0
                    DataLoader.dispatch(
                        layer_idx = layer_idx,
                        line_buffer_reshape = 0,
                        is_padding_row = is_padding_row,
                        read_mode = 0,
                        transnum = 1,
                        line_buffer_idx = dataloadermanager.line_buffer_idx,
                        src_buffer_idx = 'b',
                        bas_addr = dataloadermanager.bas_addr_cur + 4 * ky_load_num_per_cal_index - (4 if cal_idx < padding_num and (ky_load_num_per_cal_index == 1 or ky_load_num_per_cal_index == 2) else 0)
                    )
                    dataloadermanager.line_buffer_idx_switch()
                
                    # WeightLoader
                    WeightLoader.dispatch(
                        acc_reg_comp_idx = weightloadermanager.acc_reg_idx,
                        kernal_size = 0,
                        line_buffer_row_shift = 6,
                        line_buffer_idx = weightloadermanager.line_buffer_idx,
                        is_padding_col = 4,
                        weight_parall_mode = 2,
                        is_new = 0 if ky_load_num_per_cal_index == 0 else 1, # 开始新的时候覆盖
                        transnum = 24,
                        bas_addr = weightloadermanager.bas_addr_cur[2] + 24 * ky_load_num_per_cal_index , # 需要递增
                        is_bilinear_bicubic = 0,
                        offset_reg_idx = 0
                    )
                    weightloadermanager.line_buffer_idx_switch()

                # DataStorer
                DataStorer.dispatch(
                    quant_config_idx = datastorermanager.quant_config_idx,
                    pixelshuffle_out_mode = 0,
                    is_pixelshuffle = 1,
                    pooling_out_mode = 0,
                    is_pooling = 0,
                    reg_out_idx = datastorermanager.acc_reg_idx,
                    acc_mode = 3,
                    transfer_num = 0,
                    store_mode = 0,
                    stride = 18,
                    base_addr_pooling = datastorermanager.base_addr_pooling_cur,
                    base_addrs_res = datastorermanager.base_addrs_res_cur,
                    is_bicubic_add = 0,
                    is_first_or_last_row = 0,
                    is_new = 0,
                    dest_buffer_idx = 'a' # 存到a
                )
                datastorermanager.acc_reg_idx_switch()
                datastorermanager.base_addrs_res_cur += 2

                weightloadermanager.acc_reg_idx_switch()
                dataloadermanager.bas_addr_cur += 0 if cal_idx < padding_num else 4

            # 收尾
            weightloadermanager.bas_addr_cur[2] += 3*24 
            if group_level1_idx == 1 and group_level2_idx == 3: 
                datastorermanager.buffer_a_model.append({"begin":144*4*2 + 72*8 + 36*8, "end":144*4*2 + 72*8 + 36*8 + 18*8*2, "info":"c10_out", "valid":True})
            datastorermanager.buffer_b_model[-1]["valid"] = False
            quantloadermanager.quant_config_idx_switch()
            datastorermanager.quant_config_idx_switch()


    # layer 11  32*18 conv11 (128,16,3,2) 前半部分的(64,8,3) 之后加地址要跨过后半部分的
    layer_idx = 11
    for group_idx in [0]:
        quant_mode = 7
        QuantLoader.dispatch(
            quant_reg_load_idx = quantloadermanager.quant_config_idx,
            quant_mode = quant_mode,
            layer_idx = layer_idx,
            transnum = 8,
            bas_addr = quantloadermanager.bas_addr_cur
        )
        quantloadermanager.bas_addr_cur += 16 # 第二次的已经算过了

        # 初始变量设定
        cal_total_num = 18//4 + 1 # 一次算出4行 最后一次应该要特殊计算 
        ic_load_num_per_cal = 16 # 遍历ic
        ky_load_num_per_cal = 3 # 遍历ky
        load_total_num = cal_total_num * ky_load_num_per_cal * ic_load_num_per_cal
        dataloadermanager.line_buffer_load_total_num = load_total_num 
        padding_num = 1
        dataloadermanager.bas_addr_cur = 144*4*2 + 72*8 + 36*8 # 已经存了c1、c3、c5
        datastorermanager.base_addrs_res_cur = 18*2 
        datastorermanager.base_addr_pooling_cur = 0
        for cal_idx in range(cal_total_num):
            for ky_load_num_per_cal_index in range(ky_load_num_per_cal):
                for ic_load_num_per_cal_index in range(ic_load_num_per_cal):
                    # DataLoader
                    if cal_idx < padding_num and ky_load_num_per_cal_index < padding_num:
                        is_padding_row = 1
                    elif cal_idx > cal_total_num - 1 - padding_num and ky_load_num_per_cal_index > ky_load_num_per_cal - 1 - padding_num:
                        is_padding_row = 5
                    else:
                        is_padding_row = 0
                    DataLoader.dispatch(
                        layer_idx = layer_idx,
                        line_buffer_reshape = 3,
                        is_padding_row = is_padding_row,
                        read_mode = 0,
                        transnum = 4 if cal_idx < cal_total_num-1 else 2, 
                        line_buffer_idx = dataloadermanager.line_buffer_idx,
                        src_buffer_idx = 'a',
                        bas_addr = dataloadermanager.bas_addr_cur + 18 * ic_load_num_per_cal_index + ky_load_num_per_cal_index - (1 if cal_idx < padding_num and (ky_load_num_per_cal_index == 1 or ky_load_num_per_cal_index == 2) else 0) # 此时is_padding_row为0，从地址0处load出无padding的4行
                    )
                    dataloadermanager.line_buffer_idx_switch()
                
                    # WeightLoader
                    WeightLoader.dispatch(
                        acc_reg_comp_idx = weightloadermanager.acc_reg_idx,
                        kernal_size = 0,
                        line_buffer_row_shift = 0,
                        line_buffer_idx = weightloadermanager.line_buffer_idx,
                        is_padding_col = 3,
                        weight_parall_mode = 0,
                        is_new = 0 if ky_load_num_per_cal_index == 0 and ic_load_num_per_cal_index == 0 else 1, # 开始新的时候覆盖
                        transnum = 12,
                        bas_addr = weightloadermanager.bas_addr_cur[0] + 12 * (ky_load_num_per_cal_index*ic_load_num_per_cal + ic_load_num_per_cal_index), 
                        is_bilinear_bicubic = 0,
                        offset_reg_idx = 0
                    )
                    weightloadermanager.line_buffer_idx_switch()

            # DataStorer
            DataStorer.dispatch(
                quant_config_idx = datastorermanager.quant_config_idx,
                pixelshuffle_out_mode = 0,
                is_pixelshuffle = 0,
                pooling_out_mode = 0,
                is_pooling = 0,
                reg_out_idx = datastorermanager.acc_reg_idx,
                acc_mode = 2,
                transfer_num = 1 if cal_idx < cal_total_num-1 else 0,
                store_mode = 1, 
                stride = 0,
                base_addr_pooling = datastorermanager.base_addr_pooling_cur,
                base_addrs_res = datastorermanager.base_addrs_res_cur,
                is_bicubic_add = 0,
                is_first_or_last_row = 0,
                is_new = 0,
                dest_buffer_idx = 'b' # 存到b
            )
            datastorermanager.acc_reg_idx_switch()
            datastorermanager.base_addrs_res_cur += 8

            weightloadermanager.acc_reg_idx_switch()
            dataloadermanager.bas_addr_cur += 3 if cal_idx < padding_num else 4

        # 收尾
        weightloadermanager.bas_addr_cur[0] += 12*48 
        datastorermanager.buffer_b_model.append({"begin":18*2 , "end":18*2+18*2, "info":f"c11_out_group{group_idx}", "valid":True})
        datastorermanager.buffer_a_model[-1]["valid"] = False
        quantloadermanager.quant_config_idx_switch()
        datastorermanager.quant_config_idx_switch()

    # layer 12 32*18 conv12 (16,64,3,2) 
    layer_idx = 12
    group = 2 
    for group_idx in range(group):
        quant_mode = 3
        QuantLoader.dispatch(
            quant_reg_load_idx = quantloadermanager.quant_config_idx,
            quant_mode = quant_mode,
            layer_idx = layer_idx,
            transnum = 32,
            bas_addr = quantloadermanager.bas_addr_cur
        )
        quantloadermanager.bas_addr_cur += 32

        # 初始变量设定
        cal_total_num = 18 # 一次算出1行 
        ky_load_num_per_cal = 3 # 遍历ky
        load_total_num = cal_total_num * ky_load_num_per_cal
        dataloadermanager.line_buffer_load_total_num = load_total_num
        padding_num = 1
        dataloadermanager.bas_addr_cur = 0 + group_idx*18*2  # 因为存的时候两组group顺序是反的，这里权重重排即可
        datastorermanager.base_addrs_res_cur = 144*4*2 + 72*8 + 36*8 + group_idx * 4
        datastorermanager.base_addr_pooling_cur = 0
        for cal_idx in range(cal_total_num):
            for ky_load_num_per_cal_index in range(ky_load_num_per_cal):
                # DataLoader
                if cal_idx < padding_num and ky_load_num_per_cal_index < padding_num:
                    is_padding_row = 2
                elif cal_idx > cal_total_num - 1 - padding_num and ky_load_num_per_cal_index > ky_load_num_per_cal - 1 - padding_num:
                    is_padding_row = 2
                else:
                    is_padding_row = 0
                DataLoader.dispatch(
                    layer_idx = layer_idx,
                    line_buffer_reshape = 0,
                    is_padding_row = is_padding_row,
                    read_mode = 0,
                    transnum = 2, 
                    line_buffer_idx = dataloadermanager.line_buffer_idx,
                    src_buffer_idx = 'b',
                    bas_addr = dataloadermanager.bas_addr_cur + ky_load_num_per_cal_index * 2 - (2 if cal_idx < padding_num and (ky_load_num_per_cal_index == 1 or ky_load_num_per_cal_index == 2) else 0)
                )
                dataloadermanager.line_buffer_idx_switch()
            
                # WeightLoader
                WeightLoader.dispatch(
                    acc_reg_comp_idx = weightloadermanager.acc_reg_idx,
                    kernal_size = 0,
                    line_buffer_row_shift = 4,
                    line_buffer_idx = weightloadermanager.line_buffer_idx,
                    is_padding_col = 3,
                    weight_parall_mode = 2,
                    is_new = 0 if ky_load_num_per_cal_index == 0 else 1, # 开始新的时候覆盖
                    transnum = 24,
                    bas_addr = weightloadermanager.bas_addr_cur[2] + 24 * ky_load_num_per_cal_index,
                    is_bilinear_bicubic = 0,
                    offset_reg_idx = 0
                )
                weightloadermanager.line_buffer_idx_switch()

            # DataStorer
            DataStorer.dispatch(
                quant_config_idx = datastorermanager.quant_config_idx,
                pixelshuffle_out_mode = 1,
                is_pixelshuffle = 1,
                pooling_out_mode = 0,
                is_pooling = 0,
                reg_out_idx = datastorermanager.acc_reg_idx,
                acc_mode = 1,
                transfer_num = 1,
                store_mode = 3,  
                stride = 8,
                base_addr_pooling = datastorermanager.base_addr_pooling_cur,
                base_addrs_res = datastorermanager.base_addrs_res_cur,
                is_bicubic_add = 0,
                is_first_or_last_row = 0,
                is_new = 0,
                dest_buffer_idx = 'a' # 存到a
            )
            datastorermanager.acc_reg_idx_switch()
            datastorermanager.base_addrs_res_cur += 16

            weightloadermanager.acc_reg_idx_switch()
            dataloadermanager.bas_addr_cur += 0 if cal_idx < padding_num else 2

        # 收尾
        weightloadermanager.bas_addr_cur[2] += 24*3
        if group_idx == 1:
            datastorermanager.buffer_a_model.append({"begin":144*4*2+72*8+36*8 , "end":144*4*2+72*8+36*8+36*8, "info":f"c12_out", "valid":True})
            for t in range(len(datastorermanager.buffer_b_model)):
                if datastorermanager.buffer_b_model[t]["end"] - datastorermanager.buffer_b_model[t]["begin"] == 18*2:
                    datastorermanager.buffer_b_model[t]["valid"] = False
        quantloadermanager.quant_config_idx_switch()
        datastorermanager.quant_config_idx_switch()

    
    # layer 13 64*36 conv13 (32,16,3) 
    layer_idx = 13
    quant_mode = 2
    QuantLoader.dispatch(
        quant_reg_load_idx = quantloadermanager.quant_config_idx,
        quant_mode = quant_mode,
        layer_idx = layer_idx,
        transnum = 16,
        bas_addr = quantloadermanager.bas_addr_cur
    )
    quantloadermanager.bas_addr_cur += 16

    # 初始变量设定
    cal_total_num = 36 # 一次算出1行 
    ic_load_num_per_cal = 4 # 遍历ic
    ky_load_num_per_cal = 3 # 遍历ky
    load_total_num = cal_total_num * ky_load_num_per_cal * ic_load_num_per_cal
    dataloadermanager.line_buffer_load_total_num = load_total_num 
    padding_num = 1
    dataloadermanager.bas_addr_cur = 144*4*2 + 72*8
    datastorermanager.base_addrs_res_cur = 0
    datastorermanager.base_addr_pooling_cur = 0
    for cal_idx in range(cal_total_num):  
        for ky_load_num_per_cal_index in range(ky_load_num_per_cal):
            for ic_load_num_per_cal_index in range(ic_load_num_per_cal):
                # DataLoader
                if cal_idx < padding_num and ky_load_num_per_cal_index < padding_num:
                    is_padding_row = 2
                elif cal_idx > cal_total_num - 1 - padding_num and ky_load_num_per_cal_index > ky_load_num_per_cal - 1 - padding_num:
                    is_padding_row = 2
                else:
                    is_padding_row = 0
                DataLoader.dispatch(
                    layer_idx = layer_idx,
                    line_buffer_reshape = 0,
                    is_padding_row = is_padding_row,
                    read_mode = 0,
                    transnum = 4, 
                    line_buffer_idx = dataloadermanager.line_buffer_idx,
                    src_buffer_idx = 'a',
                    bas_addr = dataloadermanager.bas_addr_cur + (ic_load_num_per_cal_index * 4 if ic_load_num_per_cal_index <= 1 else 36*8 + (ic_load_num_per_cal_index-2)*4) + 
                                                                ky_load_num_per_cal_index * 8 - (8 if cal_idx < padding_num and (ky_load_num_per_cal_index == 1 or ky_load_num_per_cal_index == 2) else 0)
                )
                dataloadermanager.line_buffer_idx_switch()
            
                # WeightLoader
                WeightLoader.dispatch(
                    acc_reg_comp_idx = weightloadermanager.acc_reg_idx,
                    kernal_size = 0,
                    line_buffer_row_shift = 3,
                    line_buffer_idx = weightloadermanager.line_buffer_idx,
                    is_padding_col = 2,
                    weight_parall_mode = 1,
                    is_new = 0 if ky_load_num_per_cal_index == 0 and ic_load_num_per_cal_index == 0 else 1, # 开始新的时候覆盖
                    transnum = 24,
                    bas_addr = weightloadermanager.bas_addr_cur[1] + 24 * (ky_load_num_per_cal_index * ic_load_num_per_cal + ic_load_num_per_cal_index),
                    is_bilinear_bicubic = 0,
                    offset_reg_idx = 0
                )
                weightloadermanager.line_buffer_idx_switch()

        # DataStorer
        DataStorer.dispatch(
            quant_config_idx = datastorermanager.quant_config_idx,
            pixelshuffle_out_mode = 0,
            is_pixelshuffle = 0,
            pooling_out_mode = 0,
            is_pooling = 0,
            reg_out_idx = datastorermanager.acc_reg_idx,
            acc_mode = 1,
            transfer_num = 1,
            store_mode = 1, 
            stride = 0,
            base_addr_pooling = datastorermanager.base_addr_pooling_cur,
            base_addrs_res = datastorermanager.base_addrs_res_cur,
            is_bicubic_add = 0,
            is_first_or_last_row = 0,
            is_new = 0,
            dest_buffer_idx = 'b' # 存到b
        )
        datastorermanager.acc_reg_idx_switch()
        datastorermanager.base_addrs_res_cur += 8

        weightloadermanager.acc_reg_idx_switch()
        dataloadermanager.bas_addr_cur += 0 if cal_idx < padding_num else 8

    # 收尾
    weightloadermanager.bas_addr_cur[1] += 24*12
    if group_idx == 1:
        datastorermanager.buffer_b_model.append({"begin":0 , "end":36*8, "info":f"c13_out", "valid":True})
        for t in range(len(datastorermanager.buffer_a_model)):
            if datastorermanager.buffer_a_model[t]["end"] - datastorermanager.buffer_a_model[t]["begin"] == 36*8:
                datastorermanager.buffer_a_model[t]["valid"] = False
    quantloadermanager.quant_config_idx_switch()
    datastorermanager.quant_config_idx_switch()
      
    
    # layer 14 64*36 conv14 (16,32,3) 
    layer_idx = 14
    quant_mode = 2    

    # 初始变量设定
    cal_total_num = 36 # 一次算出1行 
    oc_load_num_per_cal = 2 # 遍历oc
    ic_load_num_per_cal = 2 # 遍历ic
    ky_load_num_per_cal = 3 # 遍历ky
    load_total_num = cal_total_num * ky_load_num_per_cal * ic_load_num_per_cal * oc_load_num_per_cal
    dataloadermanager.line_buffer_load_total_num = load_total_num 
    padding_num = 1
    dataloadermanager.bas_addr_cur = 0
    datastorermanager.base_addrs_res_cur = 144*4*2 + 72*8
    datastorermanager.base_addr_pooling_cur = 0
    for oc_load_num_per_cal_index in range(oc_load_num_per_cal):
        QuantLoader.dispatch(
            quant_reg_load_idx = quantloadermanager.quant_config_idx,
            quant_mode = quant_mode,
            layer_idx = layer_idx,
            transnum = 16,
            bas_addr = quantloadermanager.bas_addr_cur
        )
        quantloadermanager.bas_addr_cur += 16
        dataloadermanager.bas_addr_cur = 0
        datastorermanager.base_addrs_res_cur = 144*4*2 + 72*8
        datastorermanager.base_addr_pooling_cur = 0
        for cal_idx in range(cal_total_num):    
            for ky_load_num_per_cal_index in range(ky_load_num_per_cal):
                for ic_load_num_per_cal_index in range(ic_load_num_per_cal):
                    # DataLoader
                    if cal_idx < padding_num and ky_load_num_per_cal_index < padding_num:
                        is_padding_row = 2
                    elif cal_idx > cal_total_num - 1 - padding_num and ky_load_num_per_cal_index > ky_load_num_per_cal - 1 - padding_num:
                        is_padding_row = 2
                    else:
                        is_padding_row = 0
                    DataLoader.dispatch(
                        layer_idx = layer_idx,
                        line_buffer_reshape = 0,
                        is_padding_row = is_padding_row,
                        read_mode = 0,
                        transnum = 4, 
                        line_buffer_idx = dataloadermanager.line_buffer_idx,
                        src_buffer_idx = 'b',
                        bas_addr = dataloadermanager.bas_addr_cur + ic_load_num_per_cal_index * 4 + ky_load_num_per_cal_index * 8 - (8 if cal_idx < padding_num and (ky_load_num_per_cal_index == 1 or ky_load_num_per_cal_index == 2) == 1 else 0)
                    )
                    dataloadermanager.line_buffer_idx_switch()
                
                    # WeightLoader
                    WeightLoader.dispatch(
                        acc_reg_comp_idx = weightloadermanager.acc_reg_idx,
                        kernal_size = 0,
                        line_buffer_row_shift = 3,
                        line_buffer_idx = weightloadermanager.line_buffer_idx,
                        is_padding_col = 2,
                        weight_parall_mode = 1,
                        is_new = 0 if ky_load_num_per_cal_index == 0 and ic_load_num_per_cal_index == 0 else 1, # 开始新的时候覆盖
                        transnum = 24,
                        bas_addr = weightloadermanager.bas_addr_cur[1] + 24 * (oc_load_num_per_cal_index * ic_load_num_per_cal * ky_load_num_per_cal + ky_load_num_per_cal_index * ic_load_num_per_cal + ic_load_num_per_cal_index),
                        is_bilinear_bicubic = 0,
                        offset_reg_idx = 0
                    )
                    weightloadermanager.line_buffer_idx_switch()

            # DataStorer
            DataStorer.dispatch(
                quant_config_idx = datastorermanager.quant_config_idx,
                pixelshuffle_out_mode = 2,
                is_pixelshuffle = 1,
                pooling_out_mode = 0,
                is_pooling = 0,
                reg_out_idx = datastorermanager.acc_reg_idx,
                acc_mode = 1,
                transfer_num = 1,
                store_mode = 1,  # 存疑 解决：oc重排
                stride = 0,
                base_addr_pooling = datastorermanager.base_addr_pooling_cur,
                base_addrs_res = datastorermanager.base_addrs_res_cur + oc_load_num_per_cal_index * 8,
                is_bicubic_add = 0,
                is_first_or_last_row = 0,
                is_new = 0,
                dest_buffer_idx = 'a' # 存到a
            )
            datastorermanager.acc_reg_idx_switch()
            datastorermanager.base_addrs_res_cur += 16

            weightloadermanager.acc_reg_idx_switch()
            dataloadermanager.bas_addr_cur += 0 if cal_idx < padding_num else 8
        quantloadermanager.quant_config_idx_switch()
        datastorermanager.quant_config_idx_switch()

    # 收尾
    weightloadermanager.bas_addr_cur[1] += 24*12
    datastorermanager.buffer_a_model.append({"begin":144*4*2+72*8 , "end":144*4*2+72*8+72*8, "info":f"c14_out", "valid":True})
    datastorermanager.buffer_b_model[-1]["valid"] = False

    # layer 15 128*72 conv15 (16,8,3) 
    layer_idx = 15
    quant_mode = 1    

    QuantLoader.dispatch(
        quant_reg_load_idx = quantloadermanager.quant_config_idx,
        quant_mode = quant_mode,
        layer_idx = layer_idx,
        transnum = 8,
        bas_addr = quantloadermanager.bas_addr_cur
    )
    quantloadermanager.bas_addr_cur += 8

    # 初始变量设定
    cal_total_num = 72 # 一次算出1行 
    ic_load_num_per_cal = 4 # 遍历ic
    ky_load_num_per_cal = 3 # 遍历ky
    load_total_num = cal_total_num * ky_load_num_per_cal * ic_load_num_per_cal 
    dataloadermanager.line_buffer_load_total_num = load_total_num 
    padding_num = 1
    dataloadermanager.bas_addr_cur = 144*4*2
    datastorermanager.base_addrs_res_cur = 0
    datastorermanager.base_addr_pooling_cur = 0
    for cal_idx in range(cal_total_num):    
        for ky_load_num_per_cal_index in range(ky_load_num_per_cal):
            for ic_load_num_per_cal_index in range(ic_load_num_per_cal):
                # DataLoader
                if cal_idx < padding_num and ky_load_num_per_cal_index < padding_num:
                    is_padding_row = 2
                elif cal_idx > cal_total_num - 1 - padding_num and ky_load_num_per_cal_index > ky_load_num_per_cal - 1 - padding_num:
                    is_padding_row = 2
                else:
                    is_padding_row = 0
                DataLoader.dispatch(
                    layer_idx = layer_idx,
                    line_buffer_reshape = 0,
                    is_padding_row = is_padding_row,
                    read_mode = 0,
                    transnum = 4, 
                    line_buffer_idx = dataloadermanager.line_buffer_idx,
                    src_buffer_idx = 'a',
                    bas_addr = dataloadermanager.bas_addr_cur + (ic_load_num_per_cal_index * 4 if ic_load_num_per_cal_index <=1 else 72*8+(ic_load_num_per_cal_index-2)*4) 
                                + ky_load_num_per_cal_index * 8 - (8 if cal_idx < padding_num and (ky_load_num_per_cal_index == 1 or ky_load_num_per_cal_index == 2) else 0)
                )
                dataloadermanager.line_buffer_idx_switch()
            
                # WeightLoader
                WeightLoader.dispatch(
                    acc_reg_comp_idx = weightloadermanager.acc_reg_idx,
                    kernal_size = 0,
                    line_buffer_row_shift = 0,
                    line_buffer_idx = weightloadermanager.line_buffer_idx,
                    is_padding_col = 1,
                    weight_parall_mode = 0,
                    is_new = 0 if ky_load_num_per_cal_index == 0 and ic_load_num_per_cal_index == 0 else 1, # 开始新的时候覆盖
                    transnum = 12,
                    bas_addr = weightloadermanager.bas_addr_cur[0] + 12 * (ky_load_num_per_cal_index * ic_load_num_per_cal + ic_load_num_per_cal_index),
                    is_bilinear_bicubic = 0,
                    offset_reg_idx = 0
                )
                weightloadermanager.line_buffer_idx_switch()

        # DataStorer
        DataStorer.dispatch(
            quant_config_idx = datastorermanager.quant_config_idx,
            pixelshuffle_out_mode = 0,
            is_pixelshuffle = 0,
            pooling_out_mode = 0,
            is_pooling = 0,
            reg_out_idx = datastorermanager.acc_reg_idx,
            acc_mode = 1,
            transfer_num = 1,
            store_mode = 1,  
            stride = 0,
            base_addr_pooling = datastorermanager.base_addr_pooling_cur,
            base_addrs_res = datastorermanager.base_addrs_res_cur, 
            is_bicubic_add = 0,
            is_first_or_last_row = 0,
            is_new = 0,
            dest_buffer_idx = 'b' # 存到b
        )
        datastorermanager.acc_reg_idx_switch()
        datastorermanager.base_addrs_res_cur += 8

        weightloadermanager.acc_reg_idx_switch()
        dataloadermanager.bas_addr_cur += 0 if cal_idx < padding_num else 8

    # 收尾
    weightloadermanager.bas_addr_cur[0] += 12*12
    datastorermanager.buffer_b_model.append({"begin":0 , "end":72*8, "info":f"c15_out", "valid":True})
    for t in range(len(datastorermanager.buffer_a_model)):
        if datastorermanager.buffer_a_model[t]["end"] - datastorermanager.buffer_a_model[t]["begin"] == 72*8:
            datastorermanager.buffer_a_model[t]["valid"] = False
    quantloadermanager.quant_config_idx_switch()
    datastorermanager.quant_config_idx_switch()

    # layer 16 128*72 conv14 (8,16,3) 
    layer_idx = 16
    quant_mode = 6    
    # 初始变量设定
    cal_total_num = 72 # 一次算出1行 
    oc_load_num_per_cal = 2 # 遍历oc
    ic_load_num_per_cal = 2 # 遍历ic
    ky_load_num_per_cal = 3 # 遍历ky
    load_total_num = cal_total_num * ky_load_num_per_cal * ic_load_num_per_cal * oc_load_num_per_cal
    dataloadermanager.line_buffer_load_total_num = load_total_num 
    padding_num = 1
    dataloadermanager.bas_addr_cur = 0
    datastorermanager.base_addrs_res_cur = 144*4*2
    datastorermanager.base_addr_pooling_cur = 0  
    for oc_load_num_per_cal_index in range(oc_load_num_per_cal):
        # load量化参数
        QuantLoader.dispatch(
            quant_reg_load_idx = quantloadermanager.quant_config_idx,
            quant_mode = quant_mode,
            layer_idx = layer_idx,
            transnum = 8,
            bas_addr = quantloadermanager.bas_addr_cur,
        )
        quantloadermanager.bas_addr_cur += 8
        # 切oc的时候重置一下
        dataloadermanager.bas_addr_cur = 0
        datastorermanager.base_addrs_res_cur = 144*4*2
        datastorermanager.base_addr_pooling_cur = 0  
        for cal_idx in range(cal_total_num):  
            for ky_load_num_per_cal_index in range(ky_load_num_per_cal):
                for ic_load_num_per_cal_index in range(ic_load_num_per_cal):
                    # DataLoader
                    if cal_idx < padding_num and ky_load_num_per_cal_index < padding_num:
                        is_padding_row = 2
                    elif cal_idx > cal_total_num - 1 - padding_num and ky_load_num_per_cal_index > ky_load_num_per_cal - 1 - padding_num:
                        is_padding_row = 2
                    else:
                        is_padding_row = 0
                    DataLoader.dispatch(
                        layer_idx = layer_idx,
                        line_buffer_reshape = 0,
                        is_padding_row = is_padding_row,
                        read_mode = 0,
                        transnum = 4, 
                        line_buffer_idx = dataloadermanager.line_buffer_idx,
                        src_buffer_idx = 'b',
                        bas_addr = dataloadermanager.bas_addr_cur + ic_load_num_per_cal_index * 4 + ky_load_num_per_cal_index * 8 - (8 if cal_idx < padding_num and (ky_load_num_per_cal_index == 1 or ky_load_num_per_cal_index == 2) else 0)
                    )
                    dataloadermanager.line_buffer_idx_switch()
                
                    # WeightLoader
                    WeightLoader.dispatch(
                        acc_reg_comp_idx = weightloadermanager.acc_reg_idx,
                        kernal_size = 0,
                        line_buffer_row_shift = 0,
                        line_buffer_idx = weightloadermanager.line_buffer_idx,
                        is_padding_col = 1,
                        weight_parall_mode = 0,
                        is_new = 0 if ky_load_num_per_cal_index == 0 and ic_load_num_per_cal_index == 0 else 1, # 开始新的时候覆盖
                        transnum = 12,
                        bas_addr = weightloadermanager.bas_addr_cur[0] + 12 * (oc_load_num_per_cal_index * ic_load_num_per_cal * ky_load_num_per_cal + ky_load_num_per_cal_index * ic_load_num_per_cal + ic_load_num_per_cal_index),
                        is_bilinear_bicubic = 0,
                        offset_reg_idx = 0
                    )
                    weightloadermanager.line_buffer_idx_switch()

            # DataStorer
            DataStorer.dispatch(
                quant_config_idx = datastorermanager.quant_config_idx,
                pixelshuffle_out_mode = 2,
                is_pixelshuffle = 1,
                pooling_out_mode = 0,
                is_pooling = 0,
                reg_out_idx = datastorermanager.acc_reg_idx,
                acc_mode = 6,
                transfer_num = 1,
                store_mode = 2,  # 存疑 解决：oc重排
                stride = 144,  
                base_addr_pooling = datastorermanager.base_addr_pooling_cur,
                base_addrs_res = datastorermanager.base_addrs_res_cur + oc_load_num_per_cal_index * 1,
                is_bicubic_add = 0,
                is_first_or_last_row = 0,
                is_new = 0,
                dest_buffer_idx = 'a' # 存到a
            )
            datastorermanager.acc_reg_idx_switch()
            datastorermanager.base_addrs_res_cur += 2 # 一次产出了两行 oc重排时，先产出的是奇数行

            weightloadermanager.acc_reg_idx_switch()
            dataloadermanager.bas_addr_cur += 0 if cal_idx < padding_num else 8
        # 切换quant reg idx
        quantloadermanager.quant_config_idx_switch()
        datastorermanager.quant_config_idx_switch()

    # 收尾
    weightloadermanager.bas_addr_cur[0] += 12*12
    datastorermanager.buffer_a_model.append({"begin":144*4*2 , "end":144*4*2+144*4*2, "info":f"c16_out", "valid":True})
    datastorermanager.buffer_b_model[-1]["valid"] = False

    # layer 17、18 左半部分128*144 两层合并
    weight_bas_addr_cur_mark = weightloadermanager.bas_addr_cur[0] # 之后回来算右半边用
    quant_bas_addr_cur_mark = quantloadermanager.bas_addr_cur  # 之后回来算右半边用
    # layer 17 (8,4,3) 
    layer_idx = 17
    quant_mode = 0    

    QuantLoader.dispatch(
        quant_reg_load_idx = quantloadermanager.quant_config_idx,
        quant_mode = quant_mode,
        layer_idx = layer_idx,
        transnum = 4,
        bas_addr = quantloadermanager.bas_addr_cur
    )
    quantloadermanager.bas_addr_cur += 4

    # 初始变量设定
    cal_total_num = 144//2 # 一次算出2行 
    ic_load_num_per_cal = 8 # 遍历ic
    load_total_num = cal_total_num * ic_load_num_per_cal 
    dataloadermanager.line_buffer_load_total_num = load_total_num 
    padding_num = 1
    dataloadermanager.bas_addr_cur = 0
    datastorermanager.base_addrs_res_cur = 0
    datastorermanager.base_addr_pooling_cur = 0
    for cal_idx in range(cal_total_num):    
        for ic_load_num_per_cal_index in range(ic_load_num_per_cal):
            # DataLoader
            if cal_idx < padding_num:
                is_padding_row = 1
            elif cal_idx > cal_total_num - 1 - padding_num:
                is_padding_row = 5
            else:
                is_padding_row = 0
            DataLoader.dispatch(
                layer_idx = layer_idx,
                line_buffer_reshape = 0,
                is_padding_row = is_padding_row,
                read_mode = 0,
                transnum = 4, 
                line_buffer_idx = dataloadermanager.line_buffer_idx,
                src_buffer_idx = 'a',
                bas_addr = dataloadermanager.bas_addr_cur + (ic_load_num_per_cal_index * 144 if ic_load_num_per_cal_index <= 3 else 144*4*2 + (ic_load_num_per_cal_index - 4) * 144) 
            )
            dataloadermanager.line_buffer_idx_switch()
        
            # WeightLoader
            WeightLoader.dispatch(
                acc_reg_comp_idx = weightloadermanager.acc_reg_idx,
                kernal_size = 0,
                line_buffer_row_shift = 1,
                line_buffer_idx = weightloadermanager.line_buffer_idx,
                is_padding_col = 1,
                weight_parall_mode = 0,
                is_new = 0 if ic_load_num_per_cal_index == 0 else 1, # 开始新的时候覆盖
                transnum = 9,
                bas_addr = weightloadermanager.bas_addr_cur[0] + 9 * ic_load_num_per_cal_index,
                is_bilinear_bicubic = 0,
                offset_reg_idx = 0
            )
            weightloadermanager.line_buffer_idx_switch()

        # DataStorer
        DataStorer.dispatch(
            quant_config_idx = datastorermanager.quant_config_idx,
            pixelshuffle_out_mode = 0,
            is_pixelshuffle = 0,
            pooling_out_mode = 0,
            is_pooling = 0,
            reg_out_idx = datastorermanager.acc_reg_idx,
            acc_mode = 0,
            transfer_num = 1,
            store_mode = 0,  
            stride = 144,  
            base_addr_pooling = datastorermanager.base_addr_pooling_cur,
            base_addrs_res = datastorermanager.base_addrs_res_cur,
            is_bicubic_add = 0,
            is_first_or_last_row = 0,
            is_new = 0,
            dest_buffer_idx = 'b' # 存到b
        )
        datastorermanager.acc_reg_idx_switch()
        datastorermanager.base_addrs_res_cur += 2 # 一次产出了两行

        weightloadermanager.acc_reg_idx_switch()
        dataloadermanager.bas_addr_cur += 1 if cal_idx < padding_num else 2

    # 收尾
    weightloadermanager.bas_addr_cur[0] += 9*8
    datastorermanager.buffer_b_model.append({"begin":0, "end":144*4, "info":f"c17_out_left_part", "valid":True})
    # datastorermanager.buffer_b_model[-1]["valid"] = False
    quantloadermanager.quant_config_idx_switch()
    datastorermanager.quant_config_idx_switch()
    
    # layer 18 (4,1,3) 
    layer_idx = 18
    quant_mode = 0    

    QuantLoader.dispatch(
        quant_reg_load_idx = quantloadermanager.quant_config_idx,
        quant_mode = quant_mode,
        layer_idx = layer_idx,
        transnum = 1,
        bas_addr = quantloadermanager.bas_addr_cur
    )
    quantloadermanager.bas_addr_cur += 1

    # 初始变量设定
    cal_total_num = 144//2 # 一次算出2行 
    ic_load_num_per_cal = 4 # 遍历ic
    load_total_num = cal_total_num * ic_load_num_per_cal 
    dataloadermanager.line_buffer_load_total_num = load_total_num 
    padding_num = 1
    dataloadermanager.bas_addr_cur = 0
    datastorermanager.base_addrs_res_cur = 0
    datastorermanager.base_addr_pooling_cur = 0
    for cal_idx in range(cal_total_num):    
        for ic_load_num_per_cal_index in range(ic_load_num_per_cal):
            # DataLoader
            if cal_idx < padding_num:
                is_padding_row = 1
            elif cal_idx > cal_total_num - 1 - padding_num:
                is_padding_row = 5
            else:
                is_padding_row = 0
            DataLoader.dispatch(
                layer_idx = layer_idx,
                line_buffer_reshape = 0,
                is_padding_row = is_padding_row,
                read_mode = 0,
                transnum = 4, 
                line_buffer_idx = dataloadermanager.line_buffer_idx,
                src_buffer_idx = 'b',
                bas_addr = dataloadermanager.bas_addr_cur + ic_load_num_per_cal_index * 144
            )
            dataloadermanager.line_buffer_idx_switch()
        
            # WeightLoader
            WeightLoader.dispatch(
                acc_reg_comp_idx = weightloadermanager.acc_reg_idx,
                kernal_size = 0,
                line_buffer_row_shift = 1,
                line_buffer_idx = weightloadermanager.line_buffer_idx,
                is_padding_col = 1,
                weight_parall_mode = 0,
                is_new = 0 if ic_load_num_per_cal_index == 0 else 1, # 开始新的时候覆盖
                transnum = 9,
                bas_addr = weightloadermanager.bas_addr_cur[0] + 9 * ic_load_num_per_cal_index,
                is_bilinear_bicubic = 0,
                offset_reg_idx = 0
            )
            weightloadermanager.line_buffer_idx_switch()

        # DataStorer
        DataStorer.dispatch(
            quant_config_idx = datastorermanager.quant_config_idx,
            pixelshuffle_out_mode = 0,
            is_pixelshuffle = 0,
            pooling_out_mode = 0,
            is_pooling = 0,
            reg_out_idx = datastorermanager.acc_reg_idx,
            acc_mode = 0,
            transfer_num = 0, # 存疑
            store_mode = 1,  
            stride = 0,  
            base_addr_pooling = datastorermanager.base_addr_pooling_cur,
            base_addrs_res = datastorermanager.base_addrs_res_cur,
            is_bicubic_add = 0,
            is_first_or_last_row = 0,
            is_new = 1 if cal_idx % 4 == 0 else 0,
            dest_buffer_idx = 'unet_output_reg' # 存到unet_output_reg
        )
        datastorermanager.acc_reg_idx_switch()
        if cal_idx % 4 == 3:
            datastorermanager.base_addrs_res_cur += 2 # 交给reg自己判断，+2是给右半边留空间

        weightloadermanager.acc_reg_idx_switch()
        dataloadermanager.bas_addr_cur += 1 if cal_idx < padding_num else 2

    # 收尾
    weightloadermanager.bas_addr_cur[0] += 9*4
    # datastorermanager.buffer_a_model.append({"begin":0, "end":144, "info":f"c18_out_left_part", "valid":True})
    datastorermanager.buffer_b_model[-1]["valid"] = False
    quantloadermanager.quant_config_idx_switch()
    datastorermanager.quant_config_idx_switch()

    # layer 17、18 右半部分128*144 两层合并
    weightloadermanager.bas_addr_cur[0] = weight_bas_addr_cur_mark
    quantloadermanager.bas_addr_cur = quant_bas_addr_cur_mark
    # layer 17 (8,4,3) 
    layer_idx = 17
    quant_mode = 0    

    QuantLoader.dispatch(
        quant_reg_load_idx = quantloadermanager.quant_config_idx,
        quant_mode = quant_mode,
        layer_idx = layer_idx,
        transnum = 4,
        bas_addr = quantloadermanager.bas_addr_cur
    )
    quantloadermanager.bas_addr_cur += 4

    # 初始变量设定
    cal_total_num = 144//2 # 一次算出2行 
    ic_load_num_per_cal = 8 # 遍历ic
    load_total_num = cal_total_num * ic_load_num_per_cal 
    dataloadermanager.line_buffer_load_total_num = load_total_num 
    padding_num = 1
    dataloadermanager.bas_addr_cur = 0 + 144*4
    datastorermanager.base_addrs_res_cur = 0
    datastorermanager.base_addr_pooling_cur = 0
    for cal_idx in range(cal_total_num):    
        for ic_load_num_per_cal_index in range(ic_load_num_per_cal):
            # DataLoader
            if cal_idx < padding_num:
                is_padding_row = 1
            elif cal_idx > cal_total_num - 1 - padding_num:
                is_padding_row = 5
            else:
                is_padding_row = 0
            DataLoader.dispatch(
                layer_idx = layer_idx,
                line_buffer_reshape = 0,
                is_padding_row = is_padding_row,
                read_mode = 0,
                transnum = 4, 
                line_buffer_idx = dataloadermanager.line_buffer_idx,
                src_buffer_idx = 'a',
                bas_addr = dataloadermanager.bas_addr_cur + (ic_load_num_per_cal_index * 144 if ic_load_num_per_cal_index <= 3 else 144*4*2 + (ic_load_num_per_cal_index - 4) * 144) 
            )
            dataloadermanager.line_buffer_idx_switch()
        
            # WeightLoader
            WeightLoader.dispatch(
                acc_reg_comp_idx = weightloadermanager.acc_reg_idx,
                kernal_size = 0,
                line_buffer_row_shift = 1,
                line_buffer_idx = weightloadermanager.line_buffer_idx,
                is_padding_col = 1,
                weight_parall_mode = 0,
                is_new = 0 if ic_load_num_per_cal_index == 0 else 1, # 开始新的时候覆盖
                transnum = 9,
                bas_addr = weightloadermanager.bas_addr_cur[0] + 9 * ic_load_num_per_cal_index,
                is_bilinear_bicubic = 0,
                offset_reg_idx = 0
            )
            weightloadermanager.line_buffer_idx_switch()

        # DataStorer
        DataStorer.dispatch(
            quant_config_idx = datastorermanager.quant_config_idx,
            pixelshuffle_out_mode = 0,
            is_pixelshuffle = 0,
            pooling_out_mode = 0,
            is_pooling = 0,
            reg_out_idx = datastorermanager.acc_reg_idx,
            acc_mode = 0,
            transfer_num = 1,
            store_mode = 0,  
            stride = 144,  
            base_addr_pooling = datastorermanager.base_addr_pooling_cur,
            base_addrs_res = datastorermanager.base_addrs_res_cur,
            is_bicubic_add = 0,
            is_first_or_last_row = 0,
            is_new = 0,
            dest_buffer_idx = 'b' # 存到b
        )
        datastorermanager.acc_reg_idx_switch()
        datastorermanager.base_addrs_res_cur += 2 # 一次产出了两行

        weightloadermanager.acc_reg_idx_switch()
        dataloadermanager.bas_addr_cur += 1 if cal_idx < padding_num else 2

    # 收尾
    weightloadermanager.bas_addr_cur[0] += 9*8
    datastorermanager.buffer_b_model.append({"begin":0, "end":144*4, "info":f"c17_out_right_part", "valid":True})
    # datastorermanager.buffer_b_model[-1]["valid"] = False
    quantloadermanager.quant_config_idx_switch()
    datastorermanager.quant_config_idx_switch()
    
    # layer 18 (4,1,3) 
    layer_idx = 18
    quant_mode = 0    

    QuantLoader.dispatch(
        quant_reg_load_idx = quantloadermanager.quant_config_idx,
        quant_mode = quant_mode,
        layer_idx = layer_idx,
        transnum = 1,
        bas_addr = quantloadermanager.bas_addr_cur
    )
    quantloadermanager.bas_addr_cur += 1

    # 初始变量设定
    cal_total_num = 144//2 # 一次算出2行 
    ic_load_num_per_cal = 4 # 遍历ic
    load_total_num = cal_total_num * ic_load_num_per_cal 
    dataloadermanager.line_buffer_load_total_num = load_total_num 
    padding_num = 1
    dataloadermanager.bas_addr_cur = 0
    datastorermanager.base_addrs_res_cur = 1
    datastorermanager.base_addr_pooling_cur = 0
    for cal_idx in range(cal_total_num):    
        for ic_load_num_per_cal_index in range(ic_load_num_per_cal):
            # DataLoader
            if cal_idx < padding_num:
                is_padding_row = 1
            elif cal_idx > cal_total_num - 1 - padding_num:
                is_padding_row = 5
            else:
                is_padding_row = 0
            DataLoader.dispatch(
                layer_idx = layer_idx,
                line_buffer_reshape = 0,
                is_padding_row = is_padding_row,
                read_mode = 0,
                transnum = 4, 
                line_buffer_idx = dataloadermanager.line_buffer_idx,
                src_buffer_idx = 'b',
                bas_addr = dataloadermanager.bas_addr_cur + ic_load_num_per_cal_index * 144
            )
            dataloadermanager.line_buffer_idx_switch()
        
            # WeightLoader
            WeightLoader.dispatch(
                acc_reg_comp_idx = weightloadermanager.acc_reg_idx,
                kernal_size = 0,
                line_buffer_row_shift = 1,
                line_buffer_idx = weightloadermanager.line_buffer_idx,
                is_padding_col = 1,
                weight_parall_mode = 0,
                is_new = 0 if ic_load_num_per_cal_index == 0 else 1, # 开始新的时候覆盖
                transnum = 9,
                bas_addr = weightloadermanager.bas_addr_cur[0] + 9 * ic_load_num_per_cal_index,
                is_bilinear_bicubic = 0,
                offset_reg_idx = 0
            )
            weightloadermanager.line_buffer_idx_switch()

        # DataStorer
        DataStorer.dispatch(
            quant_config_idx = datastorermanager.quant_config_idx,
            pixelshuffle_out_mode = 0,
            is_pixelshuffle = 0,
            pooling_out_mode = 0,
            is_pooling = 0,
            reg_out_idx = datastorermanager.acc_reg_idx,
            acc_mode = 0,
            transfer_num = 0,
            store_mode = 1,  
            stride = 0,  
            base_addr_pooling = datastorermanager.base_addr_pooling_cur,
            base_addrs_res = datastorermanager.base_addrs_res_cur,
            is_bicubic_add = 0,
            is_first_or_last_row = 0,
            is_new = 1 if cal_idx % 4 == 0 else 0,
            dest_buffer_idx = 'unet_output_reg' # 存到unet_output_reg
        )
        datastorermanager.acc_reg_idx_switch()
        if cal_idx % 4 == 3:
            datastorermanager.base_addrs_res_cur += 2 

        weightloadermanager.acc_reg_idx_switch()
        dataloadermanager.bas_addr_cur += 1 if cal_idx < padding_num else 2

    # 收尾
    weightloadermanager.bas_addr_cur[0] += 9*4
    # datastorermanager.buffer_a_model.append({"begin":144, "end":144+144, "info":f"c18_out_right_part", "valid":True})
    for t in range(len(datastorermanager.buffer_a_model)):
        if datastorermanager.buffer_a_model[t]["end"] - datastorermanager.buffer_a_model[t]["begin"] == 144*4*2:
            datastorermanager.buffer_a_model[t]["valid"] = False
    datastorermanager.buffer_b_model[-1]["valid"] = False
    quantloadermanager.quant_config_idx_switch()
    datastorermanager.quant_config_idx_switch()


    # 添加dependency信息
    code_list = Inst.code_list
    for i in range(len(code_list)):
        code_list[i]['dependency'] = []
        code_list[i]['dest'] = 0
        code_list[i]['src1'] = 0
        code_list[i]['src2'] = 0
        code_list[i]['src3'] = 0
        code_list[i]['src4'] = 0
        if code_list[i]['op_code'] == 'OffchipDataLoader':
            # 处理dataloader依赖，但是这里可能会有指令依赖过远的问题，需要在第一层算完之后直接派遣下一个图像的unet计算
            # 这里对于fsrcnn的第一个图像块会有冗余依赖，但是在硬件角度应该没有推理速度的影响
            for d in range(i-1,-1,-1):
                if code_list[d]['op_code'] == 'DataLoader':
                    if code_list[d]['layer_idx'] == 0:
                        code_list[i]['dependency'].append(d)
                        break

            for d in range(i-1,-1,-1):
                if code_list[d]['op_code'] == 'OffchipDataLoader':
                    code_list[i]['dependency'].append(d)
                    break
        elif code_list[i]['op_code'] == 'WeightLoader':
            if code_list[i]["is_off"] == 1: # 片外
                for d in range(i-1,-1,-1):
                    if code_list[d]['op_code'] == 'QuantLoader':
                        if code_list[d]["is_off_load"] == 1:
                            code_list[i]['dependency'].append(d) # 这里没有break是考虑到scale和zp的两次传输
            else: # 片上 四种依赖，unet中只有三种
                for d in range(i-1,-1,-1):
                    if code_list[d]['op_code'] == 'DataLoader':
                        if code_list[d]["line_buffer_idx"] == code_list[i]["line_buffer_idx"]:
                            code_list[i]['dependency'].append(d)
                            break
                for d in range(i-1,-1,-1):
                    if code_list[d]['op_code'] == 'DataStorer':
                        if code_list[d]["reg_out_idx"] == code_list[i]["acc_reg_comp_idx"]:
                            code_list[i]['dependency'].append(d)
                            break
                for d in range(i-1,-1,-1):
                    if code_list[d]['op_code'] == 'WeightLoader': # 这个判断也包含了片内要等片外的权重加载完
                        code_list[i]['dependency'].append(d)
                        break
        elif code_list[i]['op_code'] == 'DataLoader':
            for d in range(i-1,-1,-1):
                if code_list[d]['op_code'] == 'WeightLoader':
                    if code_list[d]["line_buffer_idx"] == code_list[i]["line_buffer_idx"] and code_list[d]["is_off"] == 0:
                        code_list[i]['dependency'].append(d)
                        break
            if code_list[i]['layer_idx'] == 0:
                # 处理offchiploader依赖
                dataloader_count = 0
                for d in range(i-1,-1,-1):
                    if code_list[d]['op_code'] == 'DataLoader' and dataloader_count < 2:
                        dataloader_count += 1
                    if dataloader_count == 2:
                        break
                # 只有前两个需要处理
                if dataloader_count < 2:
                    for d in range(i-1,-1,-1):
                        if code_list[d]['op_code'] == 'OffchipDataLoader':
                            code_list[i]['dependency'].append(d)
                            break
            else:
                # 处理 DataStorer 依赖
                dataloader_count = 0
                dataloader_idx = []
                datastorer_count = 0
                datastorer_idx = []
                for d in range(i-1,-1,-1):
                    if code_list[d]['op_code'] == 'DataLoader' and dataloader_count < 2:
                        dataloader_count += 1
                        dataloader_idx.append(d)
                    if code_list[d]['op_code'] == 'DataStorer' and datastorer_count < 1:
                        datastorer_count += 1
                        datastorer_idx.append(d)
                    if dataloader_count == 2 and datastorer_count == 1:
                        break
                # 只有每层的前两个需要处理
                if code_list[dataloader_idx[0]]['layer_idx'] != code_list[i]['layer_idx'] or code_list[dataloader_idx[1]]['layer_idx'] != code_list[i]['layer_idx']:
                    code_list[i]['dependency'].append(datastorer_idx[0])
        elif code_list[i]['op_code'] == 'QuantLoader':
            for d in range(i-1,-1,-1):
                if code_list[d]['op_code'] == 'DataStorer':
                    if code_list[d]["quant_config_idx"] == code_list[i]["quant_reg_load_idx"]:
                        code_list[i]['dependency'].append(d)
                        break
        elif code_list[i]['op_code'] == 'DataStorer':
            for d in range(i-1,-1,-1):
                if code_list[d]['op_code'] == 'QuantLoader':
                    if code_list[d]["quant_reg_load_idx"] == code_list[i]["quant_config_idx"] and code_list[d]['is_off_load'] == 0: 
                        code_list[i]['dependency'].append(d)
                        break
            for d in range(i-1,-1,-1):
                if code_list[d]['op_code'] == 'WeightLoader':
                    if code_list[d]["acc_reg_comp_idx"] == code_list[i]["reg_out_idx"] and code_list[d]['is_off'] == 0: 
                        code_list[i]['dependency'].append(d)
                        break
            for d in range(i-1,-1,-1):
                if code_list[d]['op_code'] == 'DataStorer':
                    code_list[i]['dependency'].append(d)
                    break
    
    max_gap = 0
    max_gap_idx = 0
    max_dependency_num = 0
    for i in range(len(code_list)): 
        for t in code_list[i]['dependency']:
            if i-t > max_gap:
                max_gap = i-t
                max_gap_idx = i
        if len(code_list[i]['dependency']) > max_dependency_num:
            max_dependency_num = len(code_list[i]['dependency'])
    print(max_gap,max_gap_idx,max_dependency_num)
    
    # date = datetime.datetime.now()
    # with open("code/sd/pseudo_code_%s_with_dependency.txt" % date.strftime("%m%d%H%M"), "w") as f11:
    #     f11.writelines(list(map(lambda a: str(a) + '\n', code_list)))
    
    # 处理dest与src赋值  
    idle_reg_id = list(range(1, 8))[::-1]
    init_len = len(idle_reg_id)
    reg_used_count_max = 0 # 结果是最多用到 dest = 7

    occupy_list = []

    for i in range(len(code_list)):
        code_dict = code_list[i]
        dest_code = []
        src_code = []
        # 分配 dest 寄存器
        for one_code_num in code_dict["code_num"]:
            reg_id = idle_reg_id.pop()
            occupy_list.append([one_code_num, reg_id])
            dest_code.append(reg_id)
        assert 0 < len(dest_code) <= 2
        code_dict["dest"] = dest_code[0]

        # 观察最大要用多少个
        reg_used_count = init_len - len(idle_reg_id)
        reg_used_count_max = max(reg_used_count_max, reg_used_count)

        # 分配 src 寄存器
        for dependency_code_num in code_dict["dependency"]:
            for j in range(i - 1, -1, -1):
                if dependency_code_num in code_list[j]["code_num"]:
                    src_code.append(code_list[j]["dest"])
                    break
        assert len(src_code) <= 3
        code_dict["src1"] = src_code[0] if len(src_code) > 0 else 0
        code_dict["src2"] = src_code[1] if len(src_code) > 1 else 0
        code_dict["src3"] = src_code[2] if len(src_code) > 2 else 0

        # 假设之后2000条指令内没有依赖就可以解除占用
        for occ_pair in reversed(occupy_list): # 通过reversed和append才能使得尽可能从小的编号开始分配
            required = False
            for k in range(i + 1, min(len(code_list), i + 2000)):
                if occ_pair[0] in code_list[k]["dependency"]:
                    required = True
                    break
            if not required:  # free
                idle_reg_id.append(occ_pair[1])
                occupy_list.remove(occ_pair)     
    print(reg_used_count_max)


    # date = datetime.datetime.now()
    # with open("code/sd/pseudo_code_%s_with_dependency_and_reg_num.txt" % date.strftime("%m%d%H%M"), "w") as f11:
    #     f11.writelines(list(map(lambda a: str(a) + '\n', code_list)))   



    # 逐层打印指令用于debug
    # code_list = Inst.code_list
    # layer_idx_mark = 0
    # code_list_print = []
    # len_mark = 0
    # for i in range(len(code_list)):
    #     if code_list[i]['op_code'] == 'QuantLoader' and code_list[i]['layer_idx'] == layer_idx_mark + 1:
    #         date = datetime.datetime.now()
    #         with open(f"code/sd/debug/layer{layer_idx_mark}/pseudo_code_%s.txt" % date.strftime("%m%d%H%M"), "w") as f11:
    #             f11.writelines(list(map(lambda a: str(a) + '\n', code_list_print)))
    #         len_mark += len(code_list_print)
    #         code_list_print = []
    #         code_list_print.append(code_list[i])
    #         layer_idx_mark += 1
    #     else:
    #         code_list_print.append(code_list[i])
    #         if i == len(code_list) - 1:
    #             date = datetime.datetime.now()
    #             with open(f"code/sd/debug/layer{layer_idx_mark}/pseudo_code_%s.txt" % date.strftime("%m%d%H%M"), "w") as f11:
    #                 f11.writelines(list(map(lambda a: str(a) + '\n', code_list_print)))
    #             len_mark += len(code_list_print)
    # print(len_mark, len(code_list))

    # 检查quant mode
    # code_list = Inst.code_list
    # layer_idx_mark = 0
    # for i in range(len(code_list)):
    #     if code_list[i]['op_code'] == 'QuantLoader' and code_list[i]['layer_idx'] == layer_idx_mark:
    #         print(code_list[i]['quant_mode'])
    #         layer_idx_mark += 1 


    # code_list = Inst.code_list
    # date = datetime.datetime.now()
    # with open("code/sd/pseudo_code_%s.txt" % date.strftime("%m%d%H%M"), "w") as f11:
    #     f11.writelines(list(map(lambda a: str(a) + '\n', code_list)))

    # print("datastorermanager.buffer_a_model")
    # for t in range(len(datastorermanager.buffer_a_model)):
    #     print(datastorermanager.buffer_a_model[t])
    # print("datastorermanager.buffer_b_model")
    # for t in range(len(datastorermanager.buffer_b_model)):
    #     print(datastorermanager.buffer_b_model[t])


