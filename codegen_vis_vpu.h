/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/backend/contrib/vis/codegen_vis_vpu.h
 * \brief Implementation of vis Relay codegen.
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_VIS_CODEGEN_VIS_H_
#define TVM_RELAY_BACKEND_CONTRIB_VIS_CODEGEN_VIS_H_

#include "../../utils.h"
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>

#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/relay/op.h>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <cstring>
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/error/en.h"
#include "rapidjson/writer.h"

#include "../onchip_memory_allocator/allocator_builder.h"

#include "megrez/entry.h"
#include "megrez/types.h"
#include "megrez/tensor.h"
#include "megrez/node.h"
#include "megrez/op.h"
#include "megrez/any.h"
#include "megrez/graph.h"
#include "megrez/model_builder.h"
#include "megrez/registry.h"

#include "megrez/op/activation.h"
#include "megrez/op/add.h"
#include "megrez/op/avg_pool.h"
#include "megrez/op/clip.h"
#include "megrez/op/concat.h"
#include "megrez/op/convolution.h"
#include "megrez/op/conv_transpose.h"
#include "megrez/op/dwpw.h"
#include "megrez/op/elementwise.h"
#include "megrez/op/elu.h"
#include "megrez/op/fc.h"
#include "megrez/op/flatten.h"
#include "megrez/op/gemm.h"
#include "megrez/op/global_avg_pool.h"
#include "megrez/op/leakyrelu.h"
#include "megrez/op/matmul.h"
#include "megrez/op/max_pool.h"
#include "megrez/op/merge.h"
#include "megrez/op/merge_base.h"
#include "megrez/op/mish.h"
#include "megrez/op/mul.h"
#include "megrez/op/op_types.h"
#include "megrez/op/partition.h"
#include "megrez/op/partition_base.h"
#include "megrez/op/pool.h"
#include "megrez/op/reduce_mean.h"
#include "megrez/op/relu.h"
#include "megrez/op/relu1.h"
#include "megrez/op/relu6.h"
#include "megrez/op/relun.h"
#include "megrez/op/reshape.h"
#include "megrez/op/resize.h"
#include "megrez/op/sigmod.h"
#include "megrez/op/silu.h"
#include "megrez/op/slice.h"
#include "megrez/op/split.h"
#include "megrez/op/transpose.h"
#include "megrez/op/upsample.h"
#include "megrez/op/bn.h"
#include "megrez/op/pad.h"
#include "megrez/op/sub.h"
#include "megrez/op/sqrt.h"
#include "megrez/op/exp.h"
#include "megrez/op/tanh.h"
#include "megrez/op/softmax.h"
#include "megrez/op/matmul.h"
#include "megrez/op/expand.h"
#include "megrez/op/global_sum.h"

#include "vis_runtime.h"//todo

#define TEST_PARAM_FILE_MAX_SIZE 65536
#define TEST_TILING_FILE_MAX_SIZE 1048576

struct OpBuildArgs {
    const std::string& op_name;
    const tvm::relay::Call& call;
    const std::vector<std::string>& inputs;
};

using TensorVec = std::vector<std::string>;
using OpBuildFunc = std::function<TensorVec(const OpBuildArgs&)>;

using namespace vis::megrez;
namespace tvm {
namespace relay {
namespace contrib {



//***************************************To be modified**********************************************//
/*! \brief Attributes to store the compiler options for VIS. */
struct VisVpuConfigNode : public tvm::AttrsNode<VisVpuConfigNode> {
    std::string calibtableact;
    std::string calibtablewgt;
    int memPoolSize;
    std::string insFilename;
    std::string paramFilename;
    std::string tilingJson;
    std::string allocateJson;
    bool newSchedule;
    bool dataReuse;
    bool opAddrReuse;
    
    
    TVM_DECLARE_ATTRS(VisVpuConfigNode, "ext.attrs.VisVpuConfigNode") {
        TVM_ATTR_FIELD(calibtableact).set_default("./act_calibrate.json");
        TVM_ATTR_FIELD(calibtablewgt).set_default("./wgt_calibrate.json");
        TVM_ATTR_FIELD(memPoolSize).set_default(1024*1024);
        TVM_ATTR_FIELD(insFilename).set_default("insFile");
        TVM_ATTR_FIELD(paramFilename).set_default("paramFile");
        TVM_ATTR_FIELD(tilingJson).set_default("none");
        TVM_ATTR_FIELD(allocateJson).set_default("./mem_allo.json");
        TVM_ATTR_FIELD(newSchedule).set_default(true);
        TVM_ATTR_FIELD(dataReuse).set_default(true);
        TVM_ATTR_FIELD(opAddrReuse).set_default(true);
    }
};

class VisVpuConfig : public Attrs {
  public:
    TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(VisVpuConfig, Attrs, VisVpuConfigNode);
};

//******************************************************************************************************//

/*! \brief The VIS codegen */
class VisVpuCodegen : public MixedModeVisitor {

  public:
    /*!
    * \brief VIS Codegen which takes in Relay graph and compiles VIS loadable
    *
    * \param symbol The symbol that represents the graph being converted.
    * \param expr The Relay expression to be compiled.
    */
    VisVpuCodegen(std::string symbol, Expr expr);
    /*!
    * \brief Build the Relay graph into VIS engine graph, AKA. translating
    * to VIS APIs.
    */
    void BuildGraph();
    /*!
    * \brief Return the required params.
    * */
    Array<String> GetConstParams() const {
        // return = consts_;  // for future use
        return {};
    }

    struct GroupInfo
    {
        bool need_tiling = true;
        
        int index = -1;

        int tile_num = 0;

        // bool dont_support = false;

        std::vector<std::string> call_name_list;

        std::vector<int> call_index_list;

        std::vector<std::string> pre_process;

        std::vector<std::string> post_process;
    };

    using GroupInfoPtr = std::shared_ptr<GroupInfo>;

  private:
    void VisitExpr_(const ConstantNode* op) final;
    void VisitExpr_(const CallNode* op) final;
    void VisitExpr_(const TupleNode* op) final;
    void VisitExpr_(const TupleGetItemNode* op) final;
    void VisitExpr_(const FunctionNode* op) final;

    void Init();

    std::vector<std::string> BuildOp(const Call& call, const std::vector<std::string>& inputs);
    
    vis::megrez::ShapeType GetShapeType(const std::vector<int>& shape);

    vis::megrez::DataType Tvm2VisVpu(const DataType& d);

    template <typename ValueType = size_t>
    std::vector<ValueType> GetConcrete(const Array<PrimExpr>& vals){
        std::vector<ValueType> concrete;
        for (const auto& v : vals) {
            auto* val = v.as<IntImmNode>();
            ICHECK(val);
            concrete.push_back(val->value);
        }
        return concrete;
    }

    template <typename ValueType = size_t>
    std::vector<ValueType> GetConcrete_int(const Array<Integer>& vals){
        std::vector<ValueType> concrete;
        for (const auto& v : vals) {
            auto* val = v.as<IntImmNode>();
            ICHECK(val);
            concrete.push_back(val->value);
        }
        return concrete;
    }
    //function
    TensorVec Conv2D(const OpBuildArgs& op_build_args);
    TensorVec ElementWise_add(const OpBuildArgs& op_build_args);
    TensorVec ElementWise_div(const OpBuildArgs& op_build_args);
    TensorVec Pool2D_avg(const OpBuildArgs& op_build_args);
    TensorVec Clip(const OpBuildArgs& op_build_args);
    TensorVec Concatenate(const OpBuildArgs& op_build_args);
    TensorVec Conv2D_transpose(const OpBuildArgs& op_build_args);
    TensorVec Dense(const OpBuildArgs& op_build_args);
    TensorVec Flatten(const OpBuildArgs& op_build_args);
    TensorVec Global_Pool2D_avg(const OpBuildArgs& op_build_args);
    TensorVec Global_Pool2D_max(const OpBuildArgs& op_build_args);
    TensorVec LeakyReLU(const OpBuildArgs& op_build_args);
    TensorVec Pool2D_max(const OpBuildArgs& op_build_args);
    TensorVec Mul(const OpBuildArgs& op_build_args);
    TensorVec Mean(const OpBuildArgs& op_build_args);
    TensorVec Relu(const OpBuildArgs& op_build_args);
    TensorVec Reshape(const OpBuildArgs& op_build_args);
    TensorVec Resize(const OpBuildArgs& op_build_args);
    TensorVec Sigmoid(const OpBuildArgs& op_build_args);
    TensorVec Slice(const OpBuildArgs& op_build_args);
    TensorVec Split(const OpBuildArgs& op_build_args);
    TensorVec Transpose(const OpBuildArgs& op_build_args);
    TensorVec Pass(const OpBuildArgs& op_build_args);
	TensorVec Upsample(const OpBuildArgs& op_build_args);
	TensorVec Batch_norm(const OpBuildArgs& op_build_args);
	TensorVec Pad(const OpBuildArgs& op_build_args);
    TensorVec ElementWise_sub(const OpBuildArgs& op_build_args);
    TensorVec Sqrt(const OpBuildArgs& op_build_args);
    TensorVec Exp(const OpBuildArgs& op_build_args);
    TensorVec Tanh(const OpBuildArgs& op_build_args);
    TensorVec Softmax(const OpBuildArgs& op_build_args);
    TensorVec Matmul(const OpBuildArgs& op_build_args);
    TensorVec Unsqueeze(const OpBuildArgs& op_build_args);
    TensorVec BroadCastTo(const OpBuildArgs& op_build_args);
    TensorVec Power(const OpBuildArgs& op_build_args);
    TensorVec Squeeze(const OpBuildArgs& op_build_args);
    TensorVec Take(const OpBuildArgs& op_build_args);
    TensorVec Bmatmul(const OpBuildArgs& op_build_args);
    TensorVec Global_Sum(const OpBuildArgs& op_build_args);

    // bool DoTiling(const std::string& op_name, const std::string& last_op_name);

    std::unordered_map<std::string, TensorVec(VisVpuCodegen::*)(const OpBuildArgs&)> op_map;

    std::unordered_map<int,GroupInfoPtr> tile_sort_map;

    std::vector<std::string> tiling_tile_stream;
    std::vector<std::string> allo_tile_stream;

    std::unordered_map<std::string, std::vector<std::string>> tile_sched;

    // void VisitExpr_(const VarNode* op) final;
    // void VisitExpr_(const GlobalVarNode* op) final;
    // void VisitExpr_(const FunctionNode* op) final;
    // void VisitExpr_(const LetNode* op) final;
    // void VisitExpr_(const IfNode* op) final;
    // void VisitExpr_(const RefCreateNode* op) final;
    // void VisitExpr_(const RefReadNode* op) final;
    // void VisitExpr_(const RefWriteNode* op) final;
    // void VisitExpr_(const ConstructorNode* op) final;
    // void VisitExpr_(const MatchNode* op) final;

    /*! \brief The symbol that represents the graph. */
    std::string symbol_;
    /*! \brief The function to be serialized. */
    const Expr func_;
    /*! \brief The list of required constants. */
    Array<String> consts_;
    /*! \brief VIS options */
    VisVpuConfig config_{nullptr};
    /*! calibTable. */
    rapidjson::Document docAct;

    rapidjson::Document docWgt;

    rapidjson::Document docTiling;

    bool doTiling = false;

    bool useTiling = false;

    MemoryAllocatorBuilderPtr mem_allocator_builder_;

    bool doAllocate = false;

    bool useAllocate = false;
    /*! Map of Expr to vis tensor name. */
    std::map<Expr, std::vector<std::string>> vis_tensor_map_;

    //GraphPtr
    GraphPtr graph;

    int const_idx;

    int layer_idx;

    //build function
    std::vector<std::string> Build(const Call& call, const std::vector<std::string>& inputs);

    vis::megrez::DataType Get_quant_type(const std::string& op_name);

    vis::megrez::DataType Get_split_type(const std::string& op_name, const int index);

};

}  // namespace contrib
}  // namespace relay
}  // namespace tvm


#endif  // TVM_RELAY_BACKEND_CONTRIB_VIS_CODEGEN_VIS_H_
