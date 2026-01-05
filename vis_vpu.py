# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import logging
from .register import get_pattern_table, register_pattern_table
import numpy as np
import tvm.ir
from ...dataflow_pattern import wildcard, is_op, is_tuple, is_tuple_get_item, is_constant
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.expr import Call, Constant, Tuple, GlobalVar, Var, TupleGetItem
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.dataflow_pattern import *

#Import DLA quantization
from tvm.relay.quantize import qconfig
from tvm.relay.quantize._calibrate import calibrate
from tvm.relay.quantize.quantize import *

logger = logging.getLogger("VIS")
@transform.function_pass(opt_level=3)
class FoldSumsPass:
    """Fold consecutive add operations if they have const operand"""
    def transform_function(self, func, mod, ctx):
        class ConsecutiveSumsCallback(DFPatternCallback):
            def __init__(self):
                super(ConsecutiveSumsCallback, self).__init__()
                self.x = wildcard()
                self.y = is_constant()
                self.z = is_constant()
                self.pattern = self.y + self.x + self.z

            def callback(self, pre, post, node_map):
                return node_map[self.x][0] + (node_map[self.y][0] + node_map[self.z][0])

        return rewrite(ConsecutiveSumsCallback(), func, mod)

@transform.function_pass(opt_level=3)
class YoloPass:
    """Fold consecutive add operations if they have const operand"""
    def transform_function(self, func, mod, ctx):
        class YoloCallback(DFPatternCallback):
            def __init__(self):
                super(YoloCallback, self).__init__()
                self.x = wildcard()
                reshape_out = is_op("reshape")(self.x)
                split_out = is_op("split")(reshape_out)
                split_out_0_tmp = is_tuple_get_item(split_out, 0)
                sigmoid_0 = is_op("sigmoid")(split_out_0_tmp)
                split_out_1_tmp = is_tuple_get_item(split_out, 1)
                split_out_2_tmp = is_tuple_get_item(split_out, 2)
                sigmoid_2 = is_op("sigmoid")(split_out_2_tmp)
                out_tmp = is_tuple([sigmoid_0, split_out_1_tmp, sigmoid_2])
                out_tmp_1 = is_op("concatenate")(out_tmp)
                self.pattern = is_op("reshape")(out_tmp_1)

            def callback(self, pre, post, node_map):
                return node_map[self.x][0]

        return rewrite(YoloCallback(), func, mod)
    
global_conv_relu_fuse = True

def partition_for_vis(
        mod,
        params=None,
        calibtableact="./act_calibrate.json",
        calibtablewgt="./wgt_calibrate.json",
        memPoolSize = 1024*1024,
        insFilename = "inststream",
        paramFilename = "param",
        tilingJson = "none",##./tiling_info.json
        allocateJson = "none",##./mem_allo.json
        newSchedule = True,
        dataReuse = True,
        opAddrReuse = True,
        opt_level=4,
        conv_relu_fuse = True,
):
    """Partition the graph greedily offloading supported operators to VIS.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.
    opt_level : int
        TVM optimization level
    Returns
    -------
    mod_and_config : Tuple[Module, Dict[str, Any]]
        A tuple of 1) annotated and partitioned module and 2) "relay.ext.vis_vpu.options"
        configuration which should be given to PassContext when building.
    """

    config = {
        "calibtableact": calibtableact,
        "calibtablewgt": calibtablewgt,
        "memPoolSize": memPoolSize,
        "insFilename": insFilename,
        "paramFilename": paramFilename,
        "tilingJson": tilingJson,
        "allocateJson": allocateJson,
        "newSchedule": newSchedule,
        "dataReuse": dataReuse,
        "opAddrReuse": opAddrReuse,
    }


    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    
    global global_conv_relu_fuse
    global_conv_relu_fuse = conv_relu_fuse
    seq = tvm.transform.Sequential([

        transform.MergeComposite(pattern_table()),
        transform.AnnotateTarget("vis_vpu"),
        transform.MergeCompilerRegions(),
        transform.PartitionGraph(),

        transform.InferType(),
    ])
    with tvm.transform.PassContext(opt_level = opt_level):
        mod = seq(mod)

    return mod, config

def _register_external_op_helper(op_name, supported=True):
    @tvm.ir.register_op_attr(op_name, "target.vis_vpu")
    def _func_wrapper(args):
        return supported

    return _func_wrapper
# op
_register_external_op_helper("nn.conv2d")
_register_external_op_helper("expand_dims")
_register_external_op_helper("add")
_register_external_op_helper("divide")
_register_external_op_helper("nn.avg_pool2d")
_register_external_op_helper("nn.bias_add")
_register_external_op_helper("clip")
_register_external_op_helper("concatenate")
_register_external_op_helper("nn.conv2d_transpose")
_register_external_op_helper("nn.dense")
_register_external_op_helper("nn.batch_flatten")
_register_external_op_helper("nn.global_avg_pool2d")
_register_external_op_helper("nn.global_max_pool2d")
_register_external_op_helper("nn.adaptive_avg_pool1d")
_register_external_op_helper("nn.leaky_relu")
_register_external_op_helper("nn.max_pool2d")
_register_external_op_helper("multiply")
_register_external_op_helper("mean")
_register_external_op_helper("nn.relu")
_register_external_op_helper("reshape")
_register_external_op_helper("image.resize2d")
_register_external_op_helper("sigmoid")
_register_external_op_helper("strided_slice")
_register_external_op_helper("split")
_register_external_op_helper("transpose")
_register_external_op_helper("nn.upsampling")
_register_external_op_helper("nn.batch_norm")
_register_external_op_helper("exp")
_register_external_op_helper("log")
_register_external_op_helper("tanh")
_register_external_op_helper("erf")
_register_external_op_helper("nn.pad")
_register_external_op_helper("subtract")
_register_external_op_helper("sqrt")
_register_external_op_helper("power")
_register_external_op_helper("nn.batch_matmul")
_register_external_op_helper("nn.softmax")
_register_external_op_helper("broadcast_to")
_register_external_op_helper("squeeze")
_register_external_op_helper("take")
_register_external_op_helper("sum")



#annotation pass
_register_external_op_helper("annotation.stop_fusion")
_register_external_op_helper("relay.op.annotation.simulated_quantize")
_register_external_op_helper("annotation.cast_hint")

def make_pattern_conv2d_bias_expdims(with_bias=True):
    data = wildcard()
    weight = wildcard()

    bias = is_constant()
    conv = is_op('nn.conv2d')(data, weight)

    if with_bias:
        expand_dims = is_op('expand_dims')(bias)
        conv_out = is_op('add')(conv, expand_dims)
    else:
        conv_out = conv
    return conv_out

def make_pattern_conv2d_bias(with_bias=True, with_relu = True):
    data = wildcard()
    weight = wildcard()

    # bias = wildcard()
    bias = is_constant()
    conv = is_op('nn.conv2d')(data, weight)

    if with_bias:
        conv_out = is_op('add')(conv, bias)
    else:
        conv_out = conv
    
    if with_relu:
        conv_out = conv_out.optional(is_op("nn.relu"))
    return conv_out

# def make_pattern_conv2d_bias_relu(with_bias=True):
#     data = wildcard()
#     weight = wildcard()

#     # bias = wildcard()
#     bias = is_constant()
#     conv = is_op('nn.conv2d')(data, weight)

#     if with_bias:
#         conv_out = is_op('add')(conv, bias)
#     else:
#         conv_out = conv
    
#     relu_out = is_op('relu')(conv_out)
#     return relu_out

def make_pattern_dense_bias(with_bias=True):
    data = wildcard()
    weight = wildcard()
    bias = is_constant()
    dense = is_op("nn.dense")(data, weight)
    if with_bias:
        dense_out = is_op('add')(dense, bias)
    else:
        dense_out = dense
    return dense_out

def make_pattern_silu():
    data = wildcard()

    sigmoid = is_op("sigmoid")(data)
    simulated_quantize_1 = is_op("relay.op.annotation.simulated_quantize")(data, is_constant(), is_constant(), is_constant(), is_constant())
    simulated_quantize_2 = is_op("relay.op.annotation.simulated_quantize")(sigmoid, is_constant(), is_constant(), is_constant(), is_constant())
    mul = is_op("multiply")(simulated_quantize_1, simulated_quantize_2)
    return mul
def make_pattern_silu_simple():
    data = wildcard()

    sigmoid = is_op("sigmoid")(data)

    mul = is_op("multiply")(data, sigmoid)
    return mul

def make_pattern_convTranspose_bias(with_bias=True):
    data = wildcard()
    weight = wildcard()
    bias = is_constant()
    conv = is_op("nn.conv2d_transpose")(data, weight)
    if with_bias:
        conv_out = is_op('add')(conv, bias)
    else:
        conv_out = conv
    return conv_out

def make_pattern_batch_norm():
    data = wildcard()
    gamma = wildcard()
    beta = wildcard()
    mul = is_op('multiply')(data, gamma)
    add = is_op('add')(mul, beta)
    return add

def make_pattern_mish():
    data = wildcard()
    exp_1 = is_op("exp")(data)
    one = is_expr(relay.const(1)) | is_expr(relay.const(1.0))
    add_1 = is_op("add")(exp_1,one)
    log_1 = is_op("log")(add_1)
    tanh_1 = is_op("tanh")(log_1)
    simulated_quantize_1 = is_op("relay.op.annotation.simulated_quantize")(data, is_constant(), is_constant(), is_constant(), is_constant())
    simulated_quantize_2 = is_op("relay.op.annotation.simulated_quantize")(tanh_1, is_constant(), is_constant(), is_constant(), is_constant())
    mul = is_op("multiply")(simulated_quantize_1, simulated_quantize_2)
    return mul

def make_pattern_mish_simple():
    data = wildcard()
    exp_1 = is_op("exp")(data)
    one = is_expr(relay.const(1)) | is_expr(relay.const(1.0))
    add_1 = is_op("add")(exp_1,one)
    log_1 = is_op("log")(add_1)
    tanh_1 = is_op("tanh")(log_1)
    mul = is_op("multiply")(data, tanh_1)
    return mul

def make_pattern_gelu():
    data = wildcard()
    div_1 = is_op("divide")(data, is_expr(relay.const(1.4142135381698608)))
    erf_1 = is_op("erf")(div_1)
    one = is_expr(relay.const(1)) | is_expr(relay.const(1.0))
    add_1 = is_op("add")(erf_1,one)   
    simulated_quantize_1 = is_op("relay.op.annotation.simulated_quantize")(add_1, is_constant(), is_constant(), is_constant(), is_constant())   
    mul_1 = is_op("multiply")(data, simulated_quantize_1)
    simulated_quantize_2 = is_op("relay.op.annotation.simulated_quantize")(mul_1, is_constant(), is_constant(), is_constant(), is_constant())   
    mul_2 = is_op("multiply")(simulated_quantize_2, is_expr(relay.const(0.5)))
    return mul_2

def make_pattern_gelu_simple():
    data = wildcard()
    div_1 = is_op("divide")(data, is_expr(relay.const(1.4142135381698608)))
    erf_1 = is_op("erf")(div_1)
    one = is_expr(relay.const(1)) | is_expr(relay.const(1.0))
    add_1 = is_op("add")(erf_1,one)   
    mul_1 = is_op("multiply")(data, add_1)
    mul_2 = is_op("multiply")(mul_1, is_expr(relay.const(0.5)))
    return mul_2

@register_pattern_table("vis_vpu")
def pattern_table():
    # conv2d_bias_relu_pat = ("vis_vpu.conv2d_bias_relu", make_pattern_conv2d_bias_relu(with_bias=True))
    global global_conv_relu_fuse
    conv_relu_fuse = global_conv_relu_fuse
    # conv_relu_fuse = True
    conv2d_bias_expdims_pat = ("vis_vpu.conv2d_bias_expdims", make_pattern_conv2d_bias_expdims(with_bias=True))
    conv2d_bias_pat = ("vis_vpu.conv2d_bias", make_pattern_conv2d_bias(with_bias=True, with_relu = conv_relu_fuse))
    dense_bias_pat = ("vis_vpu.dense_bias", make_pattern_dense_bias(with_bias=True))
    silu_pat = ("vis_vpu.silu", make_pattern_silu())
    silu_simple_pat = ("vis_vpu.silu_simple", make_pattern_silu_simple())
    convTranspose_bias_pat = ("vis_vpu.convTranspose_bias", make_pattern_convTranspose_bias(with_bias=True))
    mish_pat = ("vis_vpu.mish", make_pattern_mish())
    mish_simple_pat = ("vis_vpu.mish", make_pattern_mish_simple())
    gelu_pat = ("vis_vpu.gelu", make_pattern_gelu())
    gelu_simple_pat = ("vis_vpu.gelu", make_pattern_gelu_simple())
    batch_norm_pat = ("vis_vpu.batch_norm", make_pattern_batch_norm())

    #patterns = [conv2d_bias_expdims_pat, conv2d_bias_pat, dense_bias_pat, silu_pat, silu_simple_pat,convTranspose_bias_pat,mish_pat, gelu_pat, batch_norm_pat]
    patterns = [conv2d_bias_expdims_pat, conv2d_bias_pat, dense_bias_pat, silu_pat, silu_simple_pat, convTranspose_bias_pat, mish_pat,mish_simple_pat,gelu_pat, gelu_simple_pat]
    return patterns
