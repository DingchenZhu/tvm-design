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

#include <tvm/relay/attrs/image.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/reduce.h>

#include "codegen_vis_vpu.h"
#include "megrez/registry.h"
using namespace vis::megrez;

namespace tvm {
namespace relay {
namespace contrib {

#define UNPACK_OP_BUILD_ARGS(args)           \
  const std::string& op_name = args.op_name; \
  const tvm::relay::Call& call = args.call;  \
  const std::vector<std::string>& inputs = args.inputs;

TensorVec VisVpuCodegen::Conv2D(const OpBuildArgs& op_build_args) {
  UNPACK_OP_BUILD_ARGS(op_build_args);
  // create node
  CHECK_EQ(inputs.size(), 2) << "number of inputs need to corrected";
  auto input_name = inputs.at(0);
  auto kernel_name = inputs.at(1);

  TensorPtr input_tensor = graph->FindTensor(input_name);
  TensorPtr kernel_tensor = graph->FindTensor(kernel_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, CONV_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetInputTensor(1, kernel_tensor);

  node->SetOutputTensor(0, output_tensor);

  auto kernel_shape = kernel_tensor->GetShape();
  // parse attr

  const auto* attrs = call->attrs.as<Conv2DAttrs>();  //

  CHECK(attrs) << "not Conv2DAttrs node";  //

  CHECK_EQ(attrs->data_layout, "NCHW") << "layout " << attrs->data_layout << " not supported";  //

  auto strides = GetConcrete<int>(attrs->strides);
  auto padding = GetConcrete<int>(attrs->padding);
  auto dilations = GetConcrete<int>(attrs->dilation);
  auto kernel_size = GetConcrete<int>(attrs->kernel_size);

  int groups = attrs->groups;

  // config param

  ConvOpPtr op_ptr = std::dynamic_pointer_cast<ConvOp>(node->GetOp());  //

  ConvParam& param = op_ptr->GetParam();  //

  param.pad_h0 = padding[0];
  param.pad_w0 = padding[1];
  param.pad_h1 = padding[2];
  param.pad_w1 = padding[3];
  param.kernel_h = kernel_size[0];
  param.kernel_w = kernel_size[1];
  param.stride_h = strides[0];
  param.stride_w = strides[1];
  param.dilation_h = dilations[0];
  param.dilation_w = dilations[1];
  param.input_channel = input_tensor->GetShape().dim[1];
  param.output_channel = kernel_shape.dim[0];
  param.group = groups;

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  // quant weight
  auto op_node = call->op.as<OpNode>();
  auto operator_id = op_node->name;
  std::string kernel_quant_name;
  kernel_quant_name += operator_id + "_weight_" + std::to_string(layer_idx) + "_0:in";
  kernel_tensor->SetName(kernel_quant_name);
  std::vector<float> channel_scale(kernel_shape.dim[0]);
  std::vector<int> channel_zero_point(kernel_shape.dim[0]);
  std::vector<float> channel_max(kernel_shape.dim[0]);
  std::vector<float> channel_min(kernel_shape.dim[0]);

  const rapidjson::Value& scale = docWgt[(kernel_quant_name).c_str()]["scale"];
  const rapidjson::Value& zero_point = docWgt[(kernel_quant_name).c_str()]["zero_point"];
  const rapidjson::Value& max = docWgt[(kernel_quant_name).c_str()]["max_value"];
  const rapidjson::Value& min = docWgt[(kernel_quant_name).c_str()]["min_value"];
  if (scale.Size() > 1) {
    assert(scale.Size() == kernel_shape.dim[0]);
    for (int i = 0; i < kernel_shape.dim[0]; i++) {
      channel_scale[i] = scale[i].GetFloat();

      if (zero_point.IsArray()) {
        channel_zero_point[i] = zero_point[i].GetFloat();
      } else {
        channel_zero_point[i] = zero_point.GetFloat();
      }

      channel_max[i] = max[i].GetFloat();

      channel_min[i] = min[i].GetFloat();
    }

    kernel_tensor->SetChannelScale(channel_scale.data(), kernel_shape.dim[0]);
    kernel_tensor->SetChannelZeroPoint(channel_zero_point.data(), kernel_shape.dim[0]);
    kernel_tensor->SetChannelMax(channel_max.data(), kernel_shape.dim[0]);
    kernel_tensor->SetChannelMin(channel_min.data(), kernel_shape.dim[0]);
    if (std::strcmp(docWgt[(kernel_quant_name).c_str()]["quantizer"].GetString(), "Symmetric") == 0)
      kernel_tensor->SetQuantType(kQT_Sym);
    else
      kernel_tensor->SetQuantType(kQT_Asym);
  } else {
    kernel_tensor->SetScale(docWgt[(kernel_quant_name).c_str()]["scale"][0].GetFloat());

    kernel_tensor->SetMax(docWgt[(kernel_quant_name).c_str()]["max_value"][0].GetFloat());
    kernel_tensor->SetMin(docWgt[(kernel_quant_name).c_str()]["min_value"][0].GetFloat());
    if (std::strcmp(docWgt[(kernel_quant_name).c_str()]["quantizer"].GetString(), "Symmetric") ==
        0) {
      kernel_tensor->SetQuantType(kQT_Sym);
      kernel_tensor->SetZeroPoint(docWgt[(kernel_quant_name).c_str()]["zero_point"].GetFloat());
    } else {
      kernel_tensor->SetQuantType(kQT_Asym);
      kernel_tensor->SetZeroPoint(docWgt[(kernel_quant_name).c_str()]["zero_point"][0].GetFloat());
    }
  }

  if (doTiling)
  {
    std::string tiling_name;
    tiling_name = op_name;
    int output_tile_num = docTiling[(tiling_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(tiling_name).c_str()]["output_dim_num"].GetInt();

    int output_tile_dim = docTiling[(tiling_name).c_str()]["output_tile_dims"][0].GetInt();
    std::vector<TileTensorShape> tile_shape_list;
    for (int k = 0; k < output_tile_num; k ++)
    {
      std::vector<int> tileShape;
      std::vector<int> tilePos;
      for (int i = 0; i < output_dim_num; ++i)
      {
        if (i == output_tile_dim)
        {
          tilePos.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
          tileShape.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - tilePos[i]);
        }
        else
        {
          assert(oshape[i] == docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt());
          tilePos.push_back(0);
          tileShape.push_back(oshape[i]);
        }
      }
      TileTensorShape tile_tensor_shape;
      tile_tensor_shape.shape = ShapeType(GetShapeType(tileShape));
      tile_tensor_shape.pos = ShapeType(GetShapeType(tilePos));
      tile_shape_list.push_back(tile_tensor_shape);
    }
    if (useTiling)
      RegisterTileSchema(op_name, tile_shape_list);

    int group_index = docTiling[(tiling_name).c_str()]["group_index"].GetInt();
    int call_index = docTiling[(tiling_name).c_str()]["call_index"].GetInt();
    auto find = tile_sort_map.find(group_index);
    if (find != tile_sort_map.end()) 
    {
      size_t insert = 0;
      for (; insert < tile_sort_map[group_index]->call_index_list.size(); insert++)
      {
        if (tile_sort_map[group_index]->call_index_list[insert] > call_index)
        {
          break;
        }
      }

      tile_sort_map[group_index]->call_index_list.insert(tile_sort_map[group_index]->call_index_list.begin() + insert, call_index);
      tile_sort_map[group_index]->call_name_list.insert(tile_sort_map[group_index]->call_name_list.begin() + insert, op_name);
    }
    else
    {
      GroupInfoPtr group_info = GroupInfoPtr(new GroupInfo);
      group_info->index = group_index;
      group_info->tile_num = output_tile_num;
      group_info->call_index_list.push_back(call_index);
      group_info->call_name_list.push_back(op_name);
      tile_sort_map[group_index] = group_info;
    }
  }

  if (doAllocate)
  {
      printf("***********doAllocate************\n");
      int output_tile_num = docTiling[(op_name).c_str()]["output_tile_num"].GetInt();
      int output_dim_num = docTiling[(op_name).c_str()]["output_dim_num"].GetInt();
      int output_tile_dim = docTiling[(op_name).c_str()]["output_tile_dims"][0].GetInt();

      for (int k = 0; k < output_tile_num; k++)
      {
          std::string op_input_name = input_tensor->GetName();
          std::string op_kernel_name = kernel_tensor->GetName();
          std::string op_output_name = output_tensor->GetName();
          std::string tile_name = op_name + ".tile" + std::to_string(k);
          mem_allocator_builder_->CreateTile(tile_name, 0);

          // create input tensor
          printf("***********create input tensor************\n");
          std::vector<int> inputTileShape;
          std::vector<int> inputTilePos;
          for (int i = 0; i < output_dim_num; i++)
          {
              int index = docTiling[(op_name).c_str()]["input_tile_pos"].Size() == 1 ? 0 : k;
              inputTilePos.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][0].GetInt());
              inputTileShape.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][1].GetInt() - inputTilePos[i]);
          }
          std::string tile_input_name = op_input_name;
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_input_name += "@";
              else
                  tile_input_name += ",";
              tile_input_name = tile_input_name + std::to_string(inputTilePos[i]);
          }
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_input_name += ":";
              else
                  tile_input_name += ",";
              tile_input_name = tile_input_name + std::to_string(inputTileShape[i]);
          }
          int input_bit_width = docAct[(op_input_name).c_str()]["bit_width"].GetInt();
          int input_block_type = BLOCK_TYPE_INPUT;
          mem_allocator_builder_->CreateBlock(tile_name, op_input_name, 
              tile_input_name, inputTileShape, inputTilePos,
              input_bit_width/8, input_block_type);

          // create kernel tensor
          printf("***********create kernel tensor************\n");
          std::vector<int> kernelTileShape;
          std::vector<int> kernelTilePos;
          for (int i = 0; i < 4; i++)
          {
              int index = docTiling[(op_name).c_str()]["weight_tile_pos"].Size() == 1 ? 0 : k;

              kernelTilePos.push_back(0);
              kernelTileShape.push_back(docTiling[(op_name).c_str()]["weight_tile_pos"][index][i][1].GetInt() - docTiling[(op_name).c_str()]["weight_tile_pos"][index][i][0].GetInt());
              
              if (i == 0)
              {
                  if (groups == 1)
                  {
                      kernelTileShape[i] = ((kernelTileShape[i] % 32) == 0) ? kernelTileShape[i] : (kernelTileShape[i] / 32 + 1) * 32;
                  }
                  
                  op_kernel_name = op_kernel_name + ":" + std::to_string(docTiling[(op_name).c_str()]["weight_tile_pos"][index][i][0].GetInt());
                  op_kernel_name = op_kernel_name + ":" + std::to_string(docTiling[(op_name).c_str()]["weight_tile_pos"][index][i][1].GetInt());
              }
          }
          std::string tile_kernel_name = op_kernel_name;
          for (int i = 0; i < 4; i++)
          {
              if (i==0)
                  tile_kernel_name += "@";
              else
                  tile_kernel_name += ",";
              tile_kernel_name = tile_kernel_name + std::to_string(kernelTilePos[i]);
          }
          for (int i = 0; i < 4; i++)
          {
              if (i==0)
                  tile_kernel_name += ":";
              else
                  tile_kernel_name += ",";
              tile_kernel_name = tile_kernel_name + std::to_string(kernelTileShape[i]);
          }
          int kernel_bit_width = docWgt[(kernel_tensor->GetName()).c_str()]["bit_width"].GetInt();
          int kernel_block_type = BLOCK_TYPE_INPUT;
          mem_allocator_builder_->CreateBlock(tile_name, op_kernel_name, 
              tile_kernel_name, kernelTileShape, kernelTilePos,
              kernel_bit_width/8, kernel_block_type);

          // create bias tensor
          printf("***********create bias tensor************\n");
          std::vector<int> biasTileShape;
          std::vector<int> biasTilePos;
          biasTilePos.push_back(0);
          biasTileShape.push_back(oshape[1]);
          int param_align = 352;
          if (scale.Size() > 1) 
          {
            param_align = 544;
          }
          biasTileShape[0] = ((biasTileShape[0] % 64) == 0) ? (biasTileShape[0] / 64) * param_align : (biasTileShape[0] / 64 + 1) * param_align;
          std::string op_bias_name = op_name + ":bias";
          op_bias_name = op_bias_name + ":" + std::to_string(0);
          op_bias_name = op_bias_name + ":" + std::to_string(oshape[1]);
          std::string tile_bias_name = op_bias_name;
          tile_bias_name += "@";
          tile_bias_name = tile_bias_name + std::to_string(biasTilePos[0]);
          tile_bias_name += ":";
          tile_bias_name = tile_bias_name + std::to_string(biasTileShape[0]);
          int bias_bit_width = 8;
          int bias_block_type = BLOCK_TYPE_INPUT;
          mem_allocator_builder_->CreateBlock(tile_name, op_bias_name, 
              tile_bias_name, biasTileShape, biasTilePos,
              bias_bit_width/8, bias_block_type);

          // create output tensor
          printf("***********create output tensor************\n");
          std::string tile_output_name = op_output_name;
          std::vector<int> outputTileShape;
          std::vector<int> outputTilePos;
          for (int i = 0; i < output_dim_num; i++)
          {
              outputTilePos.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
              outputTileShape.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - outputTilePos[i]);
          }
          printf("***********tile_output_name************\n");
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_output_name += "@";
              else
                  tile_output_name += ",";
              tile_output_name = tile_output_name + std::to_string(outputTilePos[i]);
          }
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_output_name += ":";
              else
                  tile_output_name += ",";
              tile_output_name = tile_output_name + std::to_string(outputTileShape[i]);
          }
          printf("***********output_bit_width************\n");
          int output_bit_width = docAct[(op_output_name).c_str()]["bit_width"].GetInt();
          printf("***********output_block_type************\n");
          int output_block_type = BLOCK_TYPE_OUTPUT;
          if (layer_idx + 1 == docTiling["network"]["call_num"].GetInt())
          {
              printf("***********BLOCK_TYPE_NET_OUTPUT************\n");
              output_block_type = BLOCK_TYPE_NET_OUTPUT;
          }
          printf("***********CreateBlock************\n");
          mem_allocator_builder_->CreateBlock(tile_name, op_output_name, 
              tile_output_name, outputTileShape, outputTilePos,
              output_bit_width/8, output_block_type);

          printf("***********************\n");

      }
  }

  return outputs;
}

TensorVec VisVpuCodegen::Batch_norm(const OpBuildArgs& op_build_args) {
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 5) << "number of inputs need to corrected";
  auto input_name = inputs.at(0);
  auto gamma_name = inputs.at(1);
  auto beta_name = inputs.at(2);
  auto mean_name = inputs.at(3);
  auto var_name = inputs.at(4);

  TensorPtr input_tensor = graph->FindTensor(input_name);
  TensorPtr gamma_tensor = graph->FindTensor(gamma_name);
  TensorPtr beta_tensor = graph->FindTensor(beta_name);
  TensorPtr mean_tensor = graph->FindTensor(mean_name);
  TensorPtr var_tensor = graph->FindTensor(var_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, BN_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetOutputTensor(0, output_tensor);

  // parse attr

  // config param

  BnOpPtr op_ptr = std::dynamic_pointer_cast<BnOp>(node->GetOp());  //

  BnParam& param = op_ptr->GetParam();  //

  int param_size = gamma_tensor->GetMemSize();

  param.scale.resize(param_size);
  param.b.resize(param_size);
  param.mean.resize(param_size);
  param.var.resize(param_size);

  std::memcpy(param.scale.data(), gamma_tensor->GetData(), param_size);
  std::memcpy(param.b.data(), beta_tensor->GetData(), param_size);
  std::memcpy(param.mean.data(), mean_tensor->GetData(), param_size);
  std::memcpy(param.var.data(), var_tensor->GetData(), param_size);

  // set shape
  auto output_shape = input_tensor->GetShape();
  output_tensor->SetShape(output_shape);

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  return outputs;
}

TensorVec VisVpuCodegen::ElementWise_add(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);
  printf("elemadd \n");
  // create node
  CHECK_EQ(inputs.size(), 2) << "number of inputs need to corrected";
  auto input_name_0 = inputs.at(0);
  auto input_name_1 = inputs.at(1);

  TensorPtr input_tensor_0 = graph->FindTensor(input_name_0);
  TensorPtr input_tensor_1 = graph->FindTensor(input_name_1);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, ADD_OP);  //

  node->SetInputTensor(0, input_tensor_0);
  node->SetInputTensor(1, input_tensor_1);
  node->SetOutputTensor(0, output_tensor);

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  if (doTiling)
  {
    std::string tiling_name;
    tiling_name = op_name;
    int output_tile_num = docTiling[(tiling_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(tiling_name).c_str()]["output_dim_num"].GetInt();

    int output_tile_dim = docTiling[(tiling_name).c_str()]["output_tile_dims"][0].GetInt();
    std::vector<TileTensorShape> tile_shape_list;
    printf("***********add************\n");
    printf("tilenum: %d ", output_tile_num);
    for (int k = 0; k < output_tile_num; k ++)
    {
      std::vector<int> tileShape;
      std::vector<int> tilePos;
      for (int i = 0; i < output_dim_num; ++i)
      {
        if (i == output_tile_dim)
        {
          tilePos.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
          tileShape.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - tilePos[i]);
        }
        else
        {
          assert(oshape[i] == docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt());
          tilePos.push_back(0);
          tileShape.push_back(oshape[i]);
        }
      }
      TileTensorShape tile_tensor_shape;
      tile_tensor_shape.shape = ShapeType(GetShapeType(tileShape));
      tile_tensor_shape.pos = ShapeType(GetShapeType(tilePos));
      tile_shape_list.push_back(tile_tensor_shape);
    }
    if (useTiling)
      RegisterTileSchema(op_name, tile_shape_list);

    int group_index = docTiling[(tiling_name).c_str()]["group_index"].GetInt();
    int call_index = docTiling[(tiling_name).c_str()]["call_index"].GetInt();
    auto find = tile_sort_map.find(group_index);
    if (find != tile_sort_map.end()) 
    {
      size_t insert = 0;
      for (; insert < tile_sort_map[group_index]->call_index_list.size(); insert++)
      {
        if (tile_sort_map[group_index]->call_index_list[insert] > call_index)
        {
          break;
        }
      }

      tile_sort_map[group_index]->call_index_list.insert(tile_sort_map[group_index]->call_index_list.begin() + insert, call_index);
      tile_sort_map[group_index]->call_name_list.insert(tile_sort_map[group_index]->call_name_list.begin() + insert, op_name);
    }
    else
    {
      GroupInfoPtr group_info = GroupInfoPtr(new GroupInfo);
      group_info->index = group_index;
      group_info->tile_num = output_tile_num;
      group_info->call_index_list.push_back(call_index);
      group_info->call_name_list.push_back(op_name);
      tile_sort_map[group_index] = group_info;
    }
  }

  if (doAllocate)
  {
    printf("***********doAllocate************\n");
    int output_tile_num = docTiling[(op_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(op_name).c_str()]["output_dim_num"].GetInt();
    // int output_tile_dim = docTiling[(op_name).c_str()]["output_tile_dims"][0].GetInt();
    std::string op_input_name_0 = input_tensor_0->GetName();
    std::string op_input_name_1 = input_tensor_1->GetName();
    std::string op_output_name = output_tensor->GetName();

    for (int k = 0; k < output_tile_num; k++)
    {
      std::string tile_name = op_name + ".tile" + std::to_string(k);
      mem_allocator_builder_->CreateTile(tile_name, 0);

      // create input tensor
      printf("***********create input0 tensor************\n");
      std::string tile_input_name_0 = op_input_name_0;
      std::vector<int> inputTileShape_0;
      std::vector<int> inputTilePos_0;
      for (int i = 0; i < output_dim_num; i++)
      {
        int index = docTiling[(op_name).c_str()]["input_tile_pos"].Size() == 1 ? 0 : k;
        inputTilePos_0.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][0].GetInt());
        inputTileShape_0.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][1].GetInt() - inputTilePos_0[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_input_name_0 += "@";
        else
          tile_input_name_0 += ",";
        tile_input_name_0 = tile_input_name_0 + std::to_string(inputTilePos_0[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_input_name_0 += ":";
        else
          tile_input_name_0 += ",";
        tile_input_name_0 = tile_input_name_0 + std::to_string(inputTileShape_0[i]);
      }
      int input_bit_width = docAct[(op_input_name_0).c_str()]["bit_width"].GetInt();
      int input_block_type = BLOCK_TYPE_INPUT;
      mem_allocator_builder_->CreateBlock(tile_name, op_input_name_0, 
          tile_input_name_0, inputTileShape_0, inputTilePos_0,
          input_bit_width/8, input_block_type);

      // create input tensor
      printf("***********create input1 tensor************\n");
      std::string tile_input_name_1 = op_input_name_1;
      vis::megrez::ShapeType input1_shape = input_tensor_1->GetShape();
      if (input1_shape.dim_num == 1)
      {
        printf("***********create bias tensor************\n");
        std::vector<int> biasTileShape;
        std::vector<int> biasTilePos;
        biasTilePos.push_back(0);
        biasTileShape.push_back(input1_shape.dim[0]);
        biasTileShape[0] = ((biasTileShape[0] % 64) == 0) ? (biasTileShape[0] / 64) * 352 : (biasTileShape[0] / 64 + 1) * 352;
        op_input_name_1 = op_input_name_1 + ":" + std::to_string(biasTilePos[0]);
        op_input_name_1 = op_input_name_1 + ":" + std::to_string(biasTilePos[0] + biasTileShape[0]);
        std::string tile_input_name_1 = op_input_name_1;
        tile_input_name_1 += "@";
        tile_input_name_1 = tile_input_name_1 + std::to_string(biasTilePos[0]);
        tile_input_name_1 += ":";
        tile_input_name_1 = tile_input_name_1 + std::to_string(biasTileShape[0]);
        int bias_bit_width = 8;
        int bias_block_type = BLOCK_TYPE_INPUT;
        mem_allocator_builder_->CreateBlock(tile_name, tile_input_name_1, 
            tile_input_name_1, biasTileShape, biasTilePos,
            bias_bit_width/8, bias_block_type);
      }
      else
      {
        std::vector<int> inputTileShape_1;
        std::vector<int> inputTilePos_1;
        for (int i = 0; i < output_dim_num; i++)
        {
          int index = docTiling[(op_name).c_str()]["input_tile_pos"].Size() == 1 ? 0 : k;
          inputTilePos_1.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][0].GetInt());
          inputTileShape_1.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][1].GetInt() - inputTilePos_1[i]);
        }
        for (int i = 0; i < output_dim_num; i++)
        {
          if (i==0)
            tile_input_name_1 += "@";
          else
            tile_input_name_1 += ",";
          tile_input_name_1 = tile_input_name_1 + std::to_string(inputTilePos_1[i]);
        }
        for (int i = 0; i < output_dim_num; i++)
        {
          if (i==0)
            tile_input_name_1 += ":";
          else
            tile_input_name_1 += ",";
          tile_input_name_1 = tile_input_name_1 + std::to_string(inputTileShape_1[i]);
        }
        int input_bit_width = docAct[(op_input_name_1).c_str()]["bit_width"].GetInt();
        int input_block_type = BLOCK_TYPE_INPUT;
        mem_allocator_builder_->CreateBlock(tile_name, op_input_name_1, 
            tile_input_name_1, inputTileShape_1, inputTilePos_1,
            input_bit_width/8, input_block_type);
      }
      

      // create output tensor
      printf("***********create output tensor************\n");
      std::string tile_output_name = op_output_name;
      std::vector<int> outputTileShape;
      std::vector<int> outputTilePos;
      for (int i = 0; i < output_dim_num; i++)
      {
        outputTilePos.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
        outputTileShape.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - outputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_output_name += "@";
        else
          tile_output_name += ",";
        tile_output_name = tile_output_name + std::to_string(outputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_output_name += ":";
        else
          tile_output_name += ",";
        tile_output_name = tile_output_name + std::to_string(outputTileShape[i]);
      }
      int output_bit_width = docAct[(op_output_name).c_str()]["bit_width"].GetInt();
      int output_block_type = BLOCK_TYPE_OUTPUT;
      if (layer_idx + 1 == docTiling["network"]["call_num"].GetInt())
      {
        output_block_type = BLOCK_TYPE_NET_OUTPUT;
      }
      mem_allocator_builder_->CreateBlock(tile_name, op_output_name, 
          tile_output_name, outputTileShape, outputTilePos,
          output_bit_width/8, output_block_type);


      if(docTiling[(op_name).c_str()]["can_addr_reuse"].GetBool())
      {
        mem_allocator_builder_->CreateReuse(tile_name, tile_input_name_0, tile_output_name, false);
      }
    }
  }

  auto input0_shape = input_tensor_0->GetShape();
  auto input1_shape = input_tensor_1->GetShape();

  /// change const input tensor shape from w to 1*1*w
  if (input_tensor_1->GetTensorType() == kTT_Const && input1_shape.Volume() != 1 &&
      input1_shape.dim_num == 1 &&
      input0_shape.dim[input0_shape.dim_num - 1] == input1_shape.dim[input1_shape.dim_num - 1]) {
    input1_shape.dim_num = input0_shape.dim_num;
    for (int i = 0; i < input0_shape.dim_num; ++i) input1_shape.dim[i] = 1;
    input1_shape.dim[input0_shape.dim_num - 1] = input0_shape.dim[input0_shape.dim_num - 1];
    input_tensor_1->SetShape(input1_shape);
  }

  if (input_tensor_0->GetTensorType() == kTT_Const && input0_shape.Volume() != 1 &&
      input0_shape.dim_num == 1 &&
      input0_shape.dim[input0_shape.dim_num - 1] == input1_shape.dim[input1_shape.dim_num - 1]) {
    input0_shape.dim_num = input1_shape.dim_num;
    for (int i = 0; i < input1_shape.dim_num; ++i) input0_shape.dim[i] = 1;
    input0_shape.dim[input1_shape.dim_num - 1] = input1_shape.dim[input1_shape.dim_num - 1];
    input_tensor_0->SetShape(input0_shape);
  }

  if (input_tensor_1->GetTensorType() != kTT_Const && input_tensor_0->GetTensorType() != kTT_Const)
    return outputs;

  if (input0_shape.dim[input0_shape.dim_num - 1] != input1_shape.dim[input1_shape.dim_num - 1])
    return outputs;

  if (input0_shape.Volume() != input1_shape.Volume()) {
    auto op_node = call->op.as<OpNode>();
    auto operator_id = op_node->name;
    std::string kernel_quant_name;
    kernel_quant_name += operator_id + "_bias_" + std::to_string(layer_idx) + "_0:in";

    auto kernel_tensor = input_tensor_1;
    if (input_tensor_0->GetTensorType() == kTT_Const) kernel_tensor = input_tensor_0;

    TensorPtr kernel_tensor_new =
        graph->CreateTensor(kernel_quant_name, kTT_Const, kernel_tensor->GetDataType());
    auto p = kernel_tensor->GetData();
    kernel_tensor_new->SetShape(kernel_tensor->GetShape());
    kernel_tensor_new->SetData(p, kernel_tensor->GetMemSize());
    if (input_tensor_0->GetTensorType() == kTT_Const)
      node->SetInputTensor(0, kernel_tensor_new);
    else
      node->SetInputTensor(1, kernel_tensor_new);

    return outputs;
  }

  auto kernel_tensor = input_tensor_1;
  if (input_tensor_0->GetTensorType() == kTT_Const) kernel_tensor = input_tensor_0;

  if (kernel_tensor->GetDataType() == kDT_Uint8 || kernel_tensor->GetDataType() == kDT_Int8)
    return outputs;
  if (kernel_tensor->GetDataType() == kDT_Uint16 || kernel_tensor->GetDataType() == kDT_Int16)
    return outputs;

  // quant weight
  auto op_node = call->op.as<OpNode>();
  auto operator_id = op_node->name;
  std::string kernel_quant_name;
  kernel_quant_name += operator_id + "_bias_" + std::to_string(layer_idx) + "_0:in";

  kernel_tensor->SetName(kernel_quant_name);
  auto kernel_shape = kernel_tensor->GetShape();

  std::vector<float> channel_scale(kernel_shape.dim[0]);
  std::vector<int> channel_zero_point(kernel_shape.dim[0]);
  std::vector<float> channel_max(kernel_shape.dim[0]);
  std::vector<float> channel_min(kernel_shape.dim[0]);

  const rapidjson::Value& scale = docWgt[(kernel_quant_name).c_str()]["scale"];
  const rapidjson::Value& zero_point = docWgt[(kernel_quant_name).c_str()]["zero_point"];
  const rapidjson::Value& max = docWgt[(kernel_quant_name).c_str()]["max_value"];
  const rapidjson::Value& min = docWgt[(kernel_quant_name).c_str()]["min_value"];

  if (scale.Size() > 1) {
    assert(scale.Size() == kernel_shape.dim[0]);
    for (int i = 0; i < kernel_shape.dim[0]; i++) {
      channel_scale[i] = scale[i].GetFloat();
      if (zero_point.IsArray()) {
        channel_zero_point[i] = zero_point[i].GetFloat();
      } else {
        channel_zero_point[i] = zero_point.GetFloat();
      }
      channel_max[i] = max[i].GetFloat();
      channel_min[i] = min[i].GetFloat();
    }
    kernel_tensor->SetDataType(input_tensor_0->GetDataType());
    kernel_tensor->SetChannelScale(channel_scale.data(), kernel_shape.dim[0]);
    kernel_tensor->SetChannelZeroPoint(channel_zero_point.data(), kernel_shape.dim[0]);
    kernel_tensor->SetChannelMax(channel_max.data(), kernel_shape.dim[0]);
    kernel_tensor->SetChannelMin(channel_min.data(), kernel_shape.dim[0]);
    if (std::strcmp(docWgt[(kernel_quant_name).c_str()]["quantizer"].GetString(), "Symmetric") == 0)
      kernel_tensor->SetQuantType(kQT_Sym);
    else
      kernel_tensor->SetQuantType(kQT_Asym);
  } else {
    kernel_tensor->SetScale(docWgt[(kernel_quant_name).c_str()]["scale"][0].GetFloat());

    kernel_tensor->SetMax(docWgt[(kernel_quant_name).c_str()]["max_value"][0].GetFloat());
    kernel_tensor->SetMin(docWgt[(kernel_quant_name).c_str()]["min_value"][0].GetFloat());
    // param.quant_mode = kQM_PerTensor;
    if (std::strcmp(docWgt[(kernel_quant_name).c_str()]["quantizer"].GetString(), "Symmetric") ==
        0) {
      kernel_tensor->SetQuantType(kQT_Sym);
      kernel_tensor->SetZeroPoint(docWgt[(kernel_quant_name).c_str()]["zero_point"].GetFloat());
    } else {
      kernel_tensor->SetQuantType(kQT_Asym);
      kernel_tensor->SetZeroPoint(docWgt[(kernel_quant_name).c_str()]["zero_point"][0].GetFloat());
    }
  }

  float* data_base = reinterpret_cast<float*>(kernel_tensor->GetData());
  const float kernel_scale = kernel_tensor->GetScale();
  const float kernel_zp = kernel_tensor->GetZeroPoint();
  if (kernel_tensor->GetQuantType() == kQT_Asym) {
    if (out_type == kDT_Uint16) {
      kernel_tensor->SetDataType(kDT_Uint16);
      std::vector<uint16_t> int_data(kernel_shape.Volume());
      for (int i = 0; i < kernel_shape.Volume(); ++i) {
        int_data[i] = std::round(data_base[i] / kernel_scale) + kernel_zp;
      }
      kernel_tensor->SetData(reinterpret_cast<uint8_t*>(int_data.data()), int_data.size() * 2);

    } else {
      kernel_tensor->SetDataType(kDT_Uint8);
      std::vector<uint8_t> int_data(kernel_shape.Volume());
      for (int i = 0; i < kernel_shape.Volume(); ++i) {
        int_data[i] = std::round(data_base[i] / kernel_scale) + kernel_zp;
      }
      kernel_tensor->SetData(int_data.data(), int_data.size());
    }
  } else if (kernel_tensor->GetQuantType() == kQT_Sym) {
    if (out_type == kDT_Int16) {
      kernel_tensor->SetDataType(kDT_Int16);
      std::vector<int16_t> int_data(kernel_shape.Volume());
      for (int i = 0; i < kernel_shape.Volume(); ++i) {
        int_data[i] = std::round(data_base[i] / kernel_scale);
      }
      kernel_tensor->SetData(reinterpret_cast<uint8_t*>(int_data.data()), int_data.size() * 2);

    } else {
      kernel_tensor->SetDataType(kDT_Int8);
      std::vector<int8_t> int_data(kernel_shape.Volume());
      for (int i = 0; i < kernel_shape.Volume(); ++i) {
        int_data[i] = std::round(data_base[i] / kernel_scale);
      }
      kernel_tensor->SetData(reinterpret_cast<uint8_t*>(int_data.data()), int_data.size());
    }
  }

  return outputs;
}

TensorVec VisVpuCodegen::ElementWise_sub(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 2) << "number of inputs need to corrected";
  auto input_name_0 = inputs.at(0);
  auto input_name_1 = inputs.at(1);

  TensorPtr input_tensor_0 = graph->FindTensor(input_name_0);
  TensorPtr input_tensor_1 = graph->FindTensor(input_name_1);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, SUB_OP);  //

  node->SetInputTensor(0, input_tensor_0);
  node->SetInputTensor(1, input_tensor_1);
  node->SetOutputTensor(0, output_tensor);

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  if (doTiling)
  {
    std::string tiling_name;
    tiling_name = op_name;
    int output_tile_num = docTiling[(tiling_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(tiling_name).c_str()]["output_dim_num"].GetInt();

    int output_tile_dim = docTiling[(tiling_name).c_str()]["output_tile_dims"][0].GetInt();
    std::vector<TileTensorShape> tile_shape_list;
    printf("***********add************\n");
    printf("tilenum: %d ", output_tile_num);
    for (int k = 0; k < output_tile_num; k ++)
    {
      std::vector<int> tileShape;
      std::vector<int> tilePos;
      for (int i = 0; i < output_dim_num; ++i)
      {
        if (i == output_tile_dim)
        {
          tilePos.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
          tileShape.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - tilePos[i]);
        }
        else
        {
          assert(oshape[i] == docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt());
          tilePos.push_back(0);
          tileShape.push_back(oshape[i]);
        }
      }
      TileTensorShape tile_tensor_shape;
      tile_tensor_shape.shape = ShapeType(GetShapeType(tileShape));
      tile_tensor_shape.pos = ShapeType(GetShapeType(tilePos));
      tile_shape_list.push_back(tile_tensor_shape);
    }
    if (useTiling)
      RegisterTileSchema(op_name, tile_shape_list);

    int group_index = docTiling[(tiling_name).c_str()]["group_index"].GetInt();
    int call_index = docTiling[(tiling_name).c_str()]["call_index"].GetInt();
    auto find = tile_sort_map.find(group_index);
    if (find != tile_sort_map.end()) 
    {
      size_t insert = 0;
      for (; insert < tile_sort_map[group_index]->call_index_list.size(); insert++)
      {
        if (tile_sort_map[group_index]->call_index_list[insert] > call_index)
        {
          break;
        }
      }

      tile_sort_map[group_index]->call_index_list.insert(tile_sort_map[group_index]->call_index_list.begin() + insert, call_index);
      tile_sort_map[group_index]->call_name_list.insert(tile_sort_map[group_index]->call_name_list.begin() + insert, op_name);
    }
    else
    {
      GroupInfoPtr group_info = GroupInfoPtr(new GroupInfo);
      group_info->index = group_index;
      group_info->tile_num = output_tile_num;
      group_info->call_index_list.push_back(call_index);
      group_info->call_name_list.push_back(op_name);
      tile_sort_map[group_index] = group_info;
    }
  }

  return outputs;
}

TensorVec VisVpuCodegen::ElementWise_div(const OpBuildArgs& op_build_args) {  //

  UNPACK_OP_BUILD_ARGS(op_build_args);
  std::cout << "inputs.size()" << inputs.size() << std::endl;
  std::cout << "op_name " << op_name << std::endl;

  // create node
  CHECK_EQ(inputs.size(), 2) << "number of inputs need to corrected";
  auto input_name_0 = inputs.at(0);
  auto input_name_1 = inputs.at(1);

  TensorPtr input_tensor_0 = graph->FindTensor(input_name_0);
  TensorPtr input_tensor_1 = graph->FindTensor(input_name_1);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, DIV_OP);  //

  node->SetInputTensor(0, input_tensor_0);
  node->SetInputTensor(1, input_tensor_1);
  node->SetOutputTensor(0, output_tensor);

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  return outputs;
}

TensorVec VisVpuCodegen::Pool2D_avg(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";
  auto input_name = inputs.at(0);

  TensorPtr input_tensor = graph->FindTensor(input_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, AVG_POOL_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetOutputTensor(0, output_tensor);

  // parse attr
  const auto* attrs = call->attrs.as<AvgPool2DAttrs>();  //

  CHECK(attrs) << "not AvgPool2DAttrs node";  //

  auto strides = GetConcrete<int>(attrs->strides);
  auto padding = GetConcrete<int>(attrs->padding);
  auto pool_size = GetConcrete<int>(attrs->pool_size);
  bool include = attrs->count_include_pad;

  // config param
  AvgPoolOpPtr op_ptr = std::dynamic_pointer_cast<AvgPoolOp>(node->GetOp());  //

  PoolParam& param = op_ptr->GetParam();

  param.pad_h0 = padding[0];
  param.pad_h1 = padding[1];
  param.pad_w0 = padding[2];
  param.pad_w1 = padding[3];
  param.kernel_h = pool_size[0];
  param.kernel_w = pool_size[1];
  param.stride_h = strides[0];
  param.stride_w = strides[1];

  op_ptr->SetIncludePad(include);

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  if (doTiling)
  {
    std::string tiling_name;
    tiling_name = op_name;
    int output_tile_num = docTiling[(tiling_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(tiling_name).c_str()]["output_dim_num"].GetInt();
    printf("***********conv************\n");
    printf("tiling_name: %s \n", (tiling_name).c_str());
    printf("tilenum: %d \n", output_tile_num);
    int output_tile_dim = docTiling[(tiling_name).c_str()]["output_tile_dims"][0].GetInt();
    std::vector<TileTensorShape> tile_shape_list;
    for (int k = 0; k < output_tile_num; k ++)
    {
      std::vector<int> tileShape;
      std::vector<int> tilePos;
      for (int i = 0; i < output_dim_num; ++i)
      {
        if (i == output_tile_dim)
        {
          tilePos.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
          tileShape.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - tilePos[i]);
        }
        else
        {
          assert(oshape[i] == docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt());
          tilePos.push_back(0);
          tileShape.push_back(oshape[i]);
        }
      }
      TileTensorShape tile_tensor_shape;
      tile_tensor_shape.shape = ShapeType(GetShapeType(tileShape));
      tile_tensor_shape.pos = ShapeType(GetShapeType(tilePos));
      tile_shape_list.push_back(tile_tensor_shape);
    }
    if (useTiling)
      RegisterTileSchema(op_name, tile_shape_list);

    int group_index = docTiling[(tiling_name).c_str()]["group_index"].GetInt();
    int call_index = docTiling[(tiling_name).c_str()]["call_index"].GetInt();
    auto find = tile_sort_map.find(group_index);
    if (find != tile_sort_map.end()) 
    {
      size_t insert = 0;
      for (; insert < tile_sort_map[group_index]->call_index_list.size(); insert++)
      {
        if (tile_sort_map[group_index]->call_index_list[insert] > call_index)
        {
          break;
        }
      }

      tile_sort_map[group_index]->call_index_list.insert(tile_sort_map[group_index]->call_index_list.begin() + insert, call_index);
      tile_sort_map[group_index]->call_name_list.insert(tile_sort_map[group_index]->call_name_list.begin() + insert, op_name);
    }
    else
    {
      GroupInfoPtr group_info = GroupInfoPtr(new GroupInfo);
      group_info->index = group_index;
      group_info->tile_num = output_tile_num;
      group_info->call_index_list.push_back(call_index);
      group_info->call_name_list.push_back(op_name);
      tile_sort_map[group_index] = group_info;
    }
  }
  if (doAllocate)
  {
    printf("***********doAllocate************\n");
    int output_tile_num = docTiling[(op_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(op_name).c_str()]["output_dim_num"].GetInt();
    // int output_tile_dim = docTiling[(op_name).c_str()]["output_tile_dims"][0].GetInt();
    std::string op_input_name = input_tensor->GetName();
    std::string op_output_name = output_tensor->GetName();

    for (int k = 0; k < output_tile_num; k++)
    {
      std::string tile_name = op_name + ".tile" + std::to_string(k);
      mem_allocator_builder_->CreateTile(tile_name, 0);

      // create input tensor
      printf("***********create input tensor************\n");
      std::string tile_input_name = op_input_name;
      std::vector<int> inputTileShape;
      std::vector<int> inputTilePos;
      for (int i = 0; i < output_dim_num; i++)
      {
        int index = docTiling[(op_name).c_str()]["input_tile_pos"].Size() == 1 ? 0 : k;
        inputTilePos.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][0].GetInt());
        inputTileShape.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][1].GetInt() - inputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_input_name += "@";
        else
          tile_input_name += ",";
        tile_input_name = tile_input_name + std::to_string(inputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_input_name += ":";
        else
          tile_input_name += ",";
        tile_input_name = tile_input_name + std::to_string(inputTileShape[i]);
      }
      int input_bit_width = docAct[(op_input_name).c_str()]["bit_width"].GetInt();
      int input_block_type = BLOCK_TYPE_INPUT;
      mem_allocator_builder_->CreateBlock(tile_name, op_input_name, 
          tile_input_name, inputTileShape, inputTilePos,
          input_bit_width/8, input_block_type);

      // create output tensor
      printf("***********create output tensor************\n");
      std::string tile_output_name = op_output_name;
      std::vector<int> outputTileShape;
      std::vector<int> outputTilePos;
      for (int i = 0; i < output_dim_num; i++)
      {
        outputTilePos.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
        outputTileShape.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - outputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_output_name += "@";
        else
          tile_output_name += ",";
        tile_output_name = tile_output_name + std::to_string(outputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_output_name += ":";
        else
          tile_output_name += ",";
        tile_output_name = tile_output_name + std::to_string(outputTileShape[i]);
      }
      int output_bit_width = docAct[(op_output_name).c_str()]["bit_width"].GetInt();
      int output_block_type = BLOCK_TYPE_OUTPUT;
      if (layer_idx + 1 == docTiling["network"]["call_num"].GetInt())
      {
        output_block_type = BLOCK_TYPE_NET_OUTPUT;
      }
      mem_allocator_builder_->CreateBlock(tile_name, op_output_name, 
          tile_output_name, outputTileShape, outputTilePos,
          output_bit_width/8, output_block_type);

    }
  }

  return outputs;
}

TensorVec VisVpuCodegen::Clip(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // parse attr
  const auto* attrs = call->attrs.as<ClipAttrs>();  //

  CHECK(attrs) << "not ClipAttrs node";  //

  double max = attrs->a_max;
  double min = attrs->a_min;
  int maxint = static_cast<int>(max);
  int minint = static_cast<int>(min);

  // clip or relun
  int op_type = 0;  // 0-clip 1-relu1 2-relu6 3-relun
  if (minint == 0) {
    if (maxint == 1) {
      op_type = 1;
    } else if (maxint == 6) {
      op_type = 2;
    } else {
      op_type = 3;
    }
  }

  // create node
  CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";
  auto input_name = inputs.at(0);

  TensorPtr input_tensor = graph->FindTensor(input_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, op_type == 0   ? CLIP_OP
                                            : op_type == 1 ? RELU1_OP
                                            : op_type == 2 ? RELU6_OP
                                                           : RELUN_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetOutputTensor(0, output_tensor);

  // config param
  if (op_type == 0) {
    ClipOpPtr op_ptr = std::dynamic_pointer_cast<ClipOp>(node->GetOp());  //
    op_ptr->SetMax(maxint);
    op_ptr->SetMin(minint);
  } else if (op_type == 3) {
    ReluNOpPtr op_ptr = std::dynamic_pointer_cast<ReluNOp>(node->GetOp());  //
    op_ptr->SetN(maxint);
  }

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  if (doTiling)
  {
    std::string tiling_name;
    tiling_name = op_name;
    int output_tile_num = docTiling[(tiling_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(tiling_name).c_str()]["output_dim_num"].GetInt();

    int output_tile_dim = docTiling[(tiling_name).c_str()]["output_tile_dims"][0].GetInt();
    std::vector<TileTensorShape> tile_shape_list;
    for (int k = 0; k < output_tile_num; k ++)
    {
      std::vector<int> tileShape;
      std::vector<int> tilePos;
      for (int i = 0; i < output_dim_num; ++i)
      {
        if (i == output_tile_dim)
        {
          tilePos.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
          tileShape.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - tilePos[i]);
        }
        else
        {
          assert(oshape[i] == docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt());
          tilePos.push_back(0);
          tileShape.push_back(oshape[i]);
        }
      }
      TileTensorShape tile_tensor_shape;
      tile_tensor_shape.shape = ShapeType(GetShapeType(tileShape));
      tile_tensor_shape.pos = ShapeType(GetShapeType(tilePos));
      tile_shape_list.push_back(tile_tensor_shape);
    }
    if (useTiling)
      RegisterTileSchema(op_name, tile_shape_list);

    int group_index = docTiling[(tiling_name).c_str()]["group_index"].GetInt();
    int call_index = docTiling[(tiling_name).c_str()]["call_index"].GetInt();
    auto find = tile_sort_map.find(group_index);
    if (find != tile_sort_map.end()) 
    {
      size_t insert = 0;
      for (; insert < tile_sort_map[group_index]->call_index_list.size(); insert++)
      {
        if (tile_sort_map[group_index]->call_index_list[insert] > call_index)
        {
          break;
        }
      }

      tile_sort_map[group_index]->call_index_list.insert(tile_sort_map[group_index]->call_index_list.begin() + insert, call_index);
      tile_sort_map[group_index]->call_name_list.insert(tile_sort_map[group_index]->call_name_list.begin() + insert, op_name);
    }
    else
    {
      GroupInfoPtr group_info = GroupInfoPtr(new GroupInfo);
      group_info->index = group_index;
      group_info->tile_num = output_tile_num;
      group_info->call_index_list.push_back(call_index);
      group_info->call_name_list.push_back(op_name);
      tile_sort_map[group_index] = group_info;
    }
  }

  if (doAllocate)
  {
    printf("***********doAllocate************\n");
    int output_tile_num = docTiling[(op_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(op_name).c_str()]["output_dim_num"].GetInt();
    // int output_tile_dim = docTiling[(op_name).c_str()]["output_tile_dims"][0].GetInt();
    std::string op_input_name = input_tensor->GetName();
    std::string op_output_name = output_tensor->GetName();

    for (int k = 0; k < output_tile_num; k++)
    {
      std::string tile_name = op_name + ".tile" + std::to_string(k);
      mem_allocator_builder_->CreateTile(tile_name, 0);

      // create input tensor
      printf("***********create input tensor************\n");
      std::string tile_input_name = op_input_name;
      std::vector<int> inputTileShape;
      std::vector<int> inputTilePos;
      for (int i = 0; i < output_dim_num; i++)
      {
        int index = docTiling[(op_name).c_str()]["input_tile_pos"].Size() == 1 ? 0 : k;
        inputTilePos.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][0].GetInt());
        inputTileShape.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][1].GetInt() - inputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_input_name += "@";
        else
          tile_input_name += ",";
        tile_input_name = tile_input_name + std::to_string(inputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_input_name += ":";
        else
          tile_input_name += ",";
        tile_input_name = tile_input_name + std::to_string(inputTileShape[i]);
      }
      int input_bit_width = docAct[(op_input_name).c_str()]["bit_width"].GetInt();
      int input_block_type = BLOCK_TYPE_INPUT;
      mem_allocator_builder_->CreateBlock(tile_name, op_input_name, 
          tile_input_name, inputTileShape, inputTilePos,
          input_bit_width/8, input_block_type);

      // create output tensor
      printf("***********create output tensor************\n");
      std::string tile_output_name = op_output_name;
      std::vector<int> outputTileShape;
      std::vector<int> outputTilePos;
      for (int i = 0; i < output_dim_num; i++)
      {
        outputTilePos.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
        outputTileShape.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - outputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_output_name += "@";
        else
          tile_output_name += ",";
        tile_output_name = tile_output_name + std::to_string(outputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_output_name += ":";
        else
          tile_output_name += ",";
        tile_output_name = tile_output_name + std::to_string(outputTileShape[i]);
      }
      int output_bit_width = docAct[(op_output_name).c_str()]["bit_width"].GetInt();
      int output_block_type = BLOCK_TYPE_OUTPUT;
      if (layer_idx + 1 == docTiling["network"]["call_num"].GetInt())
      {
        output_block_type = BLOCK_TYPE_NET_OUTPUT;
      }
      mem_allocator_builder_->CreateBlock(tile_name, op_output_name, 
          tile_output_name, outputTileShape, outputTilePos,
          output_bit_width/8, output_block_type);

      mem_allocator_builder_->CreateReuse(tile_name, tile_input_name, tile_output_name, false);

    }
  }

  return outputs;
}

TensorVec VisVpuCodegen::Concatenate(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  NodePtr node = graph->CreateNode(op_name, CONCAT_OP);  //

  for (size_t i = 0; i < inputs.size(); i++) {
    auto input_name = inputs.at(i);
    TensorPtr input_tensor = graph->FindTensor(input_name);
    node->SetInputTensor(i, input_tensor);
  }

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  node->SetOutputTensor(0, output_tensor);

  // parse attr
  const auto* attrs = call->attrs.as<ConcatenateAttrs>();  //

  CHECK(attrs) << "not ConcatenateAttrs node";  //

  int axis = attrs->axis;

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // config param
  ConcatOpPtr op_ptr = std::dynamic_pointer_cast<ConcatOp>(node->GetOp());  //
  if (axis < 0) {
    axis = oshape.size() + axis;
  }
  op_ptr->SetAxis(axis);

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  if (doTiling)
  {
    std::string tiling_name;
    tiling_name = op_name;
    int output_tile_num = docTiling[(tiling_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(tiling_name).c_str()]["output_dim_num"].GetInt();

    int output_tile_dim = docTiling[(tiling_name).c_str()]["output_tile_dims"][0].GetInt();
    std::vector<TileTensorShape> tile_shape_list;
    for (int k = 0; k < output_tile_num; k ++)
    {
      std::vector<int> tileShape;
      std::vector<int> tilePos;
      for (int i = 0; i < output_dim_num; ++i)
      {
        if (i == output_tile_dim)
        {
          tilePos.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
          tileShape.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - tilePos[i]);
        }
        else
        {
          assert(oshape[i] == docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt());
          tilePos.push_back(0);
          tileShape.push_back(oshape[i]);
        }
      }
      TileTensorShape tile_tensor_shape;
      tile_tensor_shape.shape = ShapeType(GetShapeType(tileShape));
      tile_tensor_shape.pos = ShapeType(GetShapeType(tilePos));
      tile_shape_list.push_back(tile_tensor_shape);
    }
    if (useTiling)
      RegisterTileSchema(op_name, tile_shape_list);

    int group_index = docTiling[(tiling_name).c_str()]["group_index"].GetInt();
    int call_index = docTiling[(tiling_name).c_str()]["call_index"].GetInt();
    auto find = tile_sort_map.find(group_index);
    if (find != tile_sort_map.end()) 
    {
      size_t insert = 0;
      for (; insert < tile_sort_map[group_index]->call_index_list.size(); insert++)
      {
        if (tile_sort_map[group_index]->call_index_list[insert] > call_index)
        {
          break;
        }
      }

      tile_sort_map[group_index]->call_index_list.insert(tile_sort_map[group_index]->call_index_list.begin() + insert, call_index);
      tile_sort_map[group_index]->call_name_list.insert(tile_sort_map[group_index]->call_name_list.begin() + insert, op_name);
    }
    else
    {
      GroupInfoPtr group_info = GroupInfoPtr(new GroupInfo);
      group_info->index = group_index;
      group_info->tile_num = output_tile_num;
      group_info->call_index_list.push_back(call_index);
      group_info->call_name_list.push_back(op_name);
      tile_sort_map[group_index] = group_info;
    }
  }

  if (doAllocate)
  {
    printf("***********doAllocate************\n");
    int output_tile_num = docTiling[(op_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(op_name).c_str()]["output_dim_num"].GetInt();
    // int output_tile_dim = docTiling[(op_name).c_str()]["output_tile_dims"][0].GetInt();
    std::string op_output_name = output_tensor->GetName();

    for (int k = 0; k < output_tile_num; k++)
    {
      std::string tile_name = op_name + ".tile" + std::to_string(k);
      mem_allocator_builder_->CreateTile(tile_name, 0);

      // create input tensor
      printf("***********create input tensor************\n");
      for (size_t m = 0; m < inputs.size(); m++) {
        auto op_input_name = inputs.at(m);
        
        std::string tile_input_name = op_input_name;
        std::vector<int> inputTileShape;
        std::vector<int> inputTilePos;
        for (int i = 0; i < output_dim_num; i++)
        {
          inputTilePos.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][m][i][0].GetInt());
          inputTileShape.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][m][i][1].GetInt() - inputTilePos[i]);
        }
        for (int i = 0; i < output_dim_num; i++)
        {
          if (i==0)
            tile_input_name += "@";
          else
            tile_input_name += ",";
          tile_input_name = tile_input_name + std::to_string(inputTilePos[i]);
        }
        for (int i = 0; i < output_dim_num; i++)
        {
          if (i==0)
            tile_input_name += ":";
          else
            tile_input_name += ",";
          tile_input_name = tile_input_name + std::to_string(inputTileShape[i]);
        }
        int input_bit_width = docAct[(op_input_name).c_str()]["bit_width"].GetInt();
        int input_block_type = BLOCK_TYPE_INPUT;
        mem_allocator_builder_->CreateBlock(tile_name, op_input_name, 
            tile_input_name, inputTileShape, inputTilePos,
            input_bit_width/8, input_block_type);
      }

      // create output tensor
      printf("***********create output tensor************\n");
      std::string tile_output_name = op_output_name;
      std::vector<int> outputTileShape;
      std::vector<int> outputTilePos;
      for (int i = 0; i < output_dim_num; i++)
      {
        outputTilePos.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
        outputTileShape.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - outputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_output_name += "@";
        else
          tile_output_name += ",";
        tile_output_name = tile_output_name + std::to_string(outputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_output_name += ":";
        else
          tile_output_name += ",";
        tile_output_name = tile_output_name + std::to_string(outputTileShape[i]);
      }
      int output_bit_width = docAct[(op_output_name).c_str()]["bit_width"].GetInt();
      int output_block_type = BLOCK_TYPE_OUTPUT;
      if (layer_idx + 1 == docTiling["network"]["call_num"].GetInt())
      {
        output_block_type = BLOCK_TYPE_NET_OUTPUT;
      }
      mem_allocator_builder_->CreateBlock(tile_name, op_output_name, 
          tile_output_name, outputTileShape, outputTilePos,
          output_bit_width/8, output_block_type);
    }
  }

  decltype(inputs) outputs{op_name + ":out"};

  return outputs;
}

TensorVec VisVpuCodegen::Conv2D_transpose(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 2) << "number of inputs need to corrected";
  auto input_name = inputs.at(0);
  auto kernel_name = inputs.at(1);

  TensorPtr input_tensor = graph->FindTensor(input_name);
  TensorPtr kernel_tensor = graph->FindTensor(kernel_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, CONV_TRANSPOSE_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetInputTensor(1, kernel_tensor);

  node->SetOutputTensor(0, output_tensor);

  auto kernel_shape = kernel_tensor->GetShape();

  // parse attr
  const auto* attrs = call->attrs.as<Conv2DTransposeAttrs>();  //

  CHECK(attrs) << "not Conv2DTransposeAttrs node";  //

  CHECK_EQ(attrs->groups, 1) << "grouped not supported yet";
  CHECK_EQ(attrs->data_layout, "NCHW") << "layout " << attrs->data_layout << " not supported";  //

  auto strides = GetConcrete<int>(attrs->strides);
  auto padding = GetConcrete<int>(attrs->padding);
  auto kernel_size = GetConcrete<int>(attrs->kernel_size);

  int groups = attrs->groups;

  // config param
  ConvTransposeOpPtr op_ptr = std::dynamic_pointer_cast<ConvTransposeOp>(node->GetOp());  //

  ConvTransposeParam& param = op_ptr->GetParam();  //

  param.pad_h0 = padding[0];
  param.pad_h1 = padding[1];
  param.pad_w0 = padding[2];
  param.pad_w1 = padding[3];
  param.kernel_h = kernel_size[0];
  param.kernel_w = kernel_size[1];
  param.stride_h = strides[0];
  param.stride_w = strides[1];
  param.input_channel = input_tensor->GetShape().dim[1];
  param.output_channel = kernel_shape.dim[0];
  param.group = groups;

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  // quant weight
  auto op_node = call->op.as<OpNode>();
  auto operator_id = op_node->name;
  std::string kernel_quant_name;
  kernel_quant_name += operator_id + "_weight_" + std::to_string(layer_idx) + "_0:in";
  kernel_tensor->SetName(kernel_quant_name);
  std::vector<float> channel_scale(kernel_shape.dim[0]);
  std::vector<int> channel_zero_point(kernel_shape.dim[0]);
  std::vector<float> channel_max(kernel_shape.dim[0]);
  std::vector<float> channel_min(kernel_shape.dim[0]);

  const rapidjson::Value& scale = docWgt[(kernel_quant_name).c_str()]["scale"];
  const rapidjson::Value& zero_point = docWgt[(kernel_quant_name).c_str()]["zero_point"];
  const rapidjson::Value& max = docWgt[(kernel_quant_name).c_str()]["max_value"];
  const rapidjson::Value& min = docWgt[(kernel_quant_name).c_str()]["min_value"];
  if (scale.Size() > 1) {
    assert(scale.Size() == kernel_shape.dim[0]);
    for (int i = 0; i < kernel_shape.dim[0]; i++) {
      channel_scale[i] = scale[i].GetFloat();
      if (zero_point.IsArray()) {
        channel_zero_point[i] = zero_point[i].GetFloat();
      } else {
        channel_zero_point[i] = zero_point.GetFloat();
      }
      channel_max[i] = max[i].GetFloat();
      channel_min[i] = min[i].GetFloat();
    }

    kernel_tensor->SetChannelScale(channel_scale.data(), kernel_shape.dim[0]);
    kernel_tensor->SetChannelZeroPoint(channel_zero_point.data(), kernel_shape.dim[0]);
    kernel_tensor->SetChannelMax(channel_max.data(), kernel_shape.dim[0]);
    kernel_tensor->SetChannelMin(channel_min.data(), kernel_shape.dim[0]);
    if (std::strcmp(docWgt[(kernel_quant_name).c_str()]["quantizer"].GetString(), "Symmetric") == 0)
      kernel_tensor->SetQuantType(kQT_Sym);
    else
      kernel_tensor->SetQuantType(kQT_Asym);
  } else {
    kernel_tensor->SetScale(docWgt[(kernel_quant_name).c_str()]["scale"][0].GetFloat());

    kernel_tensor->SetMax(docWgt[(kernel_quant_name).c_str()]["max_value"][0].GetFloat());
    kernel_tensor->SetMin(docWgt[(kernel_quant_name).c_str()]["min_value"][0].GetFloat());
    if (std::strcmp(docWgt[(kernel_quant_name).c_str()]["quantizer"].GetString(), "Symmetric") ==
        0) {
      kernel_tensor->SetQuantType(kQT_Sym);
      kernel_tensor->SetZeroPoint(docWgt[(kernel_quant_name).c_str()]["zero_point"].GetFloat());
    } else {
      kernel_tensor->SetQuantType(kQT_Asym);
      kernel_tensor->SetZeroPoint(docWgt[(kernel_quant_name).c_str()]["zero_point"][0].GetFloat());
    }
  }

  if (doTiling)
  {
    std::string tiling_name;
    tiling_name = op_name;
    int output_tile_num = docTiling[(tiling_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(tiling_name).c_str()]["output_dim_num"].GetInt();

    int output_tile_dim = docTiling[(tiling_name).c_str()]["output_tile_dims"][0].GetInt();
    std::vector<TileTensorShape> tile_shape_list;
    for (int k = 0; k < output_tile_num; k ++)
    {
      std::vector<int> tileShape;
      std::vector<int> tilePos;
      for (int i = 0; i < output_dim_num; ++i)
      {
        if (i == output_tile_dim)
        {
          tilePos.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
          tileShape.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - tilePos[i]);
        }
        else
        {
          assert(oshape[i] == docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt());
          tilePos.push_back(0);
          tileShape.push_back(oshape[i]);
        }
      }
      TileTensorShape tile_tensor_shape;
      tile_tensor_shape.shape = ShapeType(GetShapeType(tileShape));
      tile_tensor_shape.pos = ShapeType(GetShapeType(tilePos));
      tile_shape_list.push_back(tile_tensor_shape);
    }
    if (useTiling)
      RegisterTileSchema(op_name, tile_shape_list);

    int group_index = docTiling[(tiling_name).c_str()]["group_index"].GetInt();
    int call_index = docTiling[(tiling_name).c_str()]["call_index"].GetInt();
    auto find = tile_sort_map.find(group_index);
    if (find != tile_sort_map.end()) 
    {
      size_t insert = 0;
      for (; insert < tile_sort_map[group_index]->call_index_list.size(); insert++)
      {
        if (tile_sort_map[group_index]->call_index_list[insert] > call_index)
        {
          break;
        }
      }

      tile_sort_map[group_index]->call_index_list.insert(tile_sort_map[group_index]->call_index_list.begin() + insert, call_index);
      tile_sort_map[group_index]->call_name_list.insert(tile_sort_map[group_index]->call_name_list.begin() + insert, op_name);
    }
    else
    {
      GroupInfoPtr group_info = GroupInfoPtr(new GroupInfo);
      group_info->index = group_index;
      group_info->tile_num = output_tile_num;
      group_info->call_index_list.push_back(call_index);
      group_info->call_name_list.push_back(op_name);
      tile_sort_map[group_index] = group_info;
    }
  }

  return outputs;
}

TensorVec VisVpuCodegen::Flatten(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";  //
  auto input_name = inputs.at(0);

  TensorPtr input_tensor = graph->FindTensor(input_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, FLATTEN_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetOutputTensor(0, output_tensor);

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  if (doTiling)
  {
    std::string tiling_name;
    tiling_name = op_name;
    int output_tile_num = docTiling[(tiling_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(tiling_name).c_str()]["output_dim_num"].GetInt();

    int output_tile_dim = docTiling[(tiling_name).c_str()]["output_tile_dims"][0].GetInt();
    std::vector<TileTensorShape> tile_shape_list;
    for (int k = 0; k < output_tile_num; k ++)
    {
      std::vector<int> tileShape;
      std::vector<int> tilePos;
      for (int i = 0; i < output_dim_num; ++i)
      {
        if (i == output_tile_dim)
        {
          tilePos.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
          tileShape.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - tilePos[i]);
        }
        else
        {
          assert(oshape[i] == docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt());
          tilePos.push_back(0);
          tileShape.push_back(oshape[i]);
        }
      }
      TileTensorShape tile_tensor_shape;
      tile_tensor_shape.shape = ShapeType(GetShapeType(tileShape));
      tile_tensor_shape.pos = ShapeType(GetShapeType(tilePos));
      tile_shape_list.push_back(tile_tensor_shape);
    }
    if (useTiling)
      RegisterTileSchema(op_name, tile_shape_list);

    int group_index = docTiling[(tiling_name).c_str()]["group_index"].GetInt();
    int call_index = docTiling[(tiling_name).c_str()]["call_index"].GetInt();
    auto find = tile_sort_map.find(group_index);
    if (find != tile_sort_map.end()) 
    {
      size_t insert = 0;
      for (; insert < tile_sort_map[group_index]->call_index_list.size(); insert++)
      {
        if (tile_sort_map[group_index]->call_index_list[insert] > call_index)
        {
          break;
        }
      }

      tile_sort_map[group_index]->call_index_list.insert(tile_sort_map[group_index]->call_index_list.begin() + insert, call_index);
      tile_sort_map[group_index]->call_name_list.insert(tile_sort_map[group_index]->call_name_list.begin() + insert, op_name);
    }
    else
    {
      GroupInfoPtr group_info = GroupInfoPtr(new GroupInfo);
      group_info->index = group_index;
      group_info->tile_num = output_tile_num;
      group_info->call_index_list.push_back(call_index);
      group_info->call_name_list.push_back(op_name);
      tile_sort_map[group_index] = group_info;
    }
  }

  if (doAllocate)
  {
    printf("***********doAllocate************\n");
    int output_tile_num = docTiling[(op_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(op_name).c_str()]["output_dim_num"].GetInt();
    int input_dim_num = docTiling[(op_name).c_str()]["input_dim_num"].GetInt();
    // int output_tile_dim = docTiling[(op_name).c_str()]["output_tile_dims"][0].GetInt();
    std::string op_input_name = input_tensor->GetName();
    std::string op_output_name = output_tensor->GetName();

    for (int k = 0; k < output_tile_num; k++)
    {
      std::string tile_name = op_name + ".tile" + std::to_string(k);
      mem_allocator_builder_->CreateTile(tile_name, 0);

      // create input tensor
      printf("***********create input tensor************\n");
      std::string tile_input_name = op_input_name;
      std::vector<int> inputTileShape;
      std::vector<int> inputTilePos;
      for (int i = 0; i < input_dim_num; i++)
      {
        int index = docTiling[(op_name).c_str()]["input_tile_pos"].Size() == 1 ? 0 : k;
        inputTilePos.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][0].GetInt());
        inputTileShape.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][1].GetInt() - inputTilePos[i]);
      }
      for (int i = 0; i < input_dim_num; i++)
      {
        if (i==0)
          tile_input_name += "@";
        else
          tile_input_name += ",";
        tile_input_name = tile_input_name + std::to_string(inputTilePos[i]);
      }
      for (int i = 0; i < input_dim_num; i++)
      {
        if (i==0)
          tile_input_name += ":";
        else
          tile_input_name += ",";
        tile_input_name = tile_input_name + std::to_string(inputTileShape[i]);
      }
      int input_bit_width = docAct[(op_input_name).c_str()]["bit_width"].GetInt();
      int input_block_type = BLOCK_TYPE_INPUT;
      mem_allocator_builder_->CreateBlock(tile_name, op_input_name, 
          tile_input_name, inputTileShape, inputTilePos,
          input_bit_width/8, input_block_type);

      // create output tensor
      printf("***********create output tensor************\n");
      std::string tile_output_name = op_output_name;
      std::vector<int> outputTileShape;
      std::vector<int> outputTilePos;
      for (int i = 0; i < output_dim_num; i++)
      {
        outputTilePos.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
        outputTileShape.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - outputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_output_name += "@";
        else
          tile_output_name += ",";
        tile_output_name = tile_output_name + std::to_string(outputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_output_name += ":";
        else
          tile_output_name += ",";
        tile_output_name = tile_output_name + std::to_string(outputTileShape[i]);
      }
      int output_bit_width = docAct[(op_output_name).c_str()]["bit_width"].GetInt();
      int output_block_type = BLOCK_TYPE_OUTPUT;
      if (layer_idx + 1 == docTiling["network"]["call_num"].GetInt())
      {
        output_block_type = BLOCK_TYPE_NET_OUTPUT;
      }
      mem_allocator_builder_->CreateBlock(tile_name, op_output_name, 
          tile_output_name, outputTileShape, outputTilePos,
          output_bit_width/8, output_block_type);
      
      mem_allocator_builder_->CreateReuse(tile_name, tile_input_name, tile_output_name, true);

    }
  }

  return outputs;
}

TensorVec VisVpuCodegen::Global_Pool2D_avg(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";
  auto input_name = inputs.at(0);

  TensorPtr input_tensor = graph->FindTensor(input_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, GLOBAL_AVG_POOL_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetOutputTensor(0, output_tensor);
  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  if (doTiling)
  {
    std::string tiling_name;
    tiling_name = op_name;
    int output_tile_num = docTiling[(tiling_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(tiling_name).c_str()]["output_dim_num"].GetInt();

    int output_tile_dim = docTiling[(tiling_name).c_str()]["output_tile_dims"][0].GetInt();
    std::vector<TileTensorShape> tile_shape_list;
    std::vector<TileTensorShape> tile_shape_list_cast0;
    std::vector<TileTensorShape> tile_shape_list_cast1;
    for (int k = 0; k < output_tile_num; k ++)
    {
      std::vector<int> tileShape;
      std::vector<int> tilePos;
      std::vector<int> tileShapeInput;
      std::vector<int> tilePosInput;
      for (int i = 0; i < output_dim_num; ++i)
      {
        if (i == output_tile_dim)
        {
          tilePos.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
          tileShape.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - tilePos[i]);
          tilePosInput.push_back(docTiling[(tiling_name).c_str()]["input_tile_pos"][k][i][0].GetInt());
          tileShapeInput.push_back(docTiling[(tiling_name).c_str()]["input_tile_pos"][k][i][1].GetInt() - tilePos[i]);
        }
        else
        {
          assert(oshape[i] == (docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - tilePos[i]));
          tilePos.push_back(0);
          tileShape.push_back(oshape[i]);
          tilePosInput.push_back(0);
          tileShapeInput.push_back(docTiling[(tiling_name).c_str()]["input_tile_pos"][k][i][1].GetInt() - tilePos[i]);
        }
      }
      TileTensorShape tile_tensor_shape_cast0;
      tile_tensor_shape_cast0.shape = ShapeType(GetShapeType(tileShapeInput));
      tile_tensor_shape_cast0.pos = ShapeType(GetShapeType(tilePosInput));
      tile_shape_list_cast0.push_back(tile_tensor_shape_cast0);

      TileTensorShape tile_tensor_shape;
      tile_tensor_shape.shape = ShapeType(GetShapeType(tileShape));
      tile_tensor_shape.pos = ShapeType(GetShapeType(tilePos));
      tile_shape_list.push_back(tile_tensor_shape);
      
      TileTensorShape tile_tensor_shape_cast1;
      tile_tensor_shape_cast1.shape = ShapeType(GetShapeType(tileShape));
      tile_tensor_shape_cast1.pos = ShapeType(GetShapeType(tilePos));
      tile_shape_list_cast1.push_back(tile_tensor_shape_cast1);
    }
    if (useTiling)
    {
      RegisterTileSchema(op_name + "_cast0", tile_shape_list_cast0);
      RegisterTileSchema(op_name, tile_shape_list);
      RegisterTileSchema(op_name + "_cast1", tile_shape_list_cast1);
    }

    int group_index = docTiling[(tiling_name).c_str()]["group_index"].GetInt();
    int call_index = docTiling[(tiling_name).c_str()]["call_index"].GetInt();
    auto find = tile_sort_map.find(group_index);
    if (find != tile_sort_map.end()) 
    {
      size_t insert = 0;
      for (; insert < tile_sort_map[group_index]->call_index_list.size(); insert++)
      {
        if (tile_sort_map[group_index]->call_index_list[insert] > call_index)
        {
          break;
        }
      }
      tile_sort_map[group_index]->call_index_list.insert(tile_sort_map[group_index]->call_index_list.begin() + insert, call_index);
      tile_sort_map[group_index]->call_name_list.insert(tile_sort_map[group_index]->call_name_list.begin() + insert, op_name + "_cast1");

      tile_sort_map[group_index]->call_index_list.insert(tile_sort_map[group_index]->call_index_list.begin() + insert, call_index);
      tile_sort_map[group_index]->call_name_list.insert(tile_sort_map[group_index]->call_name_list.begin() + insert, op_name);

      tile_sort_map[group_index]->call_index_list.insert(tile_sort_map[group_index]->call_index_list.begin() + insert, call_index);
      tile_sort_map[group_index]->call_name_list.insert(tile_sort_map[group_index]->call_name_list.begin() + insert, op_name + "_cast0");
    }
    else
    {
      GroupInfoPtr group_info = GroupInfoPtr(new GroupInfo);
      group_info->index = group_index;
      group_info->tile_num = output_tile_num;

      group_info->call_index_list.push_back(call_index);
      group_info->call_name_list.push_back(op_name + "_cast0");

      group_info->call_index_list.push_back(call_index);
      group_info->call_name_list.push_back(op_name);

      group_info->call_index_list.push_back(call_index);
      group_info->call_name_list.push_back(op_name + "_cast1");

      tile_sort_map[group_index] = group_info;
    }
  }

  if (doAllocate)
  {
      printf("***********global ave doAllocate************\n");
      std::string op_input_name = input_tensor->GetName();
      std::string op_output_name = output_tensor->GetName();

      int output_tile_num = docTiling[(op_name).c_str()]["output_tile_num"].GetInt();
      int output_dim_num = docTiling[(op_name).c_str()]["output_dim_num"].GetInt();
      if (1)
      {
        for(int k = 0; k < output_tile_num; k++)
        {
          std::string tile_name = op_name + "_cast0.tile" + std::to_string(k);
          mem_allocator_builder_->CreateTile(tile_name, 0);
          std::string pre_input_name = input_tensor->GetName();
          std::string pre_output_name = op_name + "_cast0";
          op_input_name = pre_output_name;
          printf("***********create pre input tensor************\n");
          std::vector<int> inputTileShape;
          std::vector<int> inputTilePos;
          for (int i = 0; i < output_dim_num; i++)
          {
            int index = docTiling[(op_name).c_str()]["input_tile_pos"].Size() == 1 ? 0 : k;
            inputTilePos.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][0].GetInt());
            inputTileShape.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][1].GetInt() - inputTilePos[i]);
          }

          std::string tile_input_name = pre_input_name;
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_input_name += "@";
              else
                  tile_input_name += ",";
              tile_input_name = tile_input_name + std::to_string(inputTilePos[i]);
          }
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_input_name += ":";
              else
                  tile_input_name += ",";
              tile_input_name = tile_input_name + std::to_string(inputTileShape[i]);
          }
          int input_bit_width = 8;
          int input_block_type = BLOCK_TYPE_INPUT;
          mem_allocator_builder_->CreateBlock(tile_name, pre_input_name, 
              tile_input_name, inputTileShape, inputTilePos,
              input_bit_width/8, input_block_type);
          
          // std::cout << tile_input_name << " " << inputTileShape[2] << " " << inputTileShape[3] << std::endl;

          // create temp tensor
          std::string temp_name = tile_name + "@temp";
          std::vector<int> tempTileShape;
          std::vector<int> tempputTilePos;
          tempputTilePos.push_back(0);
          int input_len = 1;
          int output_len = 1;
          for (int i = 0; i < output_dim_num; i++)
          {
              input_len *= inputTileShape[i];
              output_len *= inputTileShape[i];
          }
          int align_size = 16*1024;
          int len = 0;
          if((input_len + output_len*2) % align_size == 0)
          {
            len = (input_len + output_len*2);
          }
          else
          {
            len = ((input_len + output_len*2) / align_size + 1) * align_size;
          }
          tempTileShape.push_back(len);
          mem_allocator_builder_->CreateBlock(tile_name, temp_name, 
              temp_name, tempTileShape, tempputTilePos,
              1, BLOCK_TYPE_TEMP);

          // create output tensor
          printf("***********create pre output tensor************\n");
          std::string tile_output_name = pre_output_name;
          std::vector<int> outputTileShape;
          std::vector<int> outputTilePos;
          for (int i = 0; i < output_dim_num; i++)
          {
            int index = docTiling[(op_name).c_str()]["input_tile_pos"].Size() == 1 ? 0 : k;
            outputTilePos.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][0].GetInt());
            outputTileShape.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][1].GetInt() - inputTilePos[i]);
          }
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_output_name += "@";
              else
                  tile_output_name += ",";
              tile_output_name = tile_output_name + std::to_string(outputTilePos[i]);
          }
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_output_name += ":";
              else
                  tile_output_name += ",";
              tile_output_name = tile_output_name + std::to_string(outputTileShape[i]);
          }
          int output_bit_width = 16;
          int output_block_type = BLOCK_TYPE_OUTPUT;
          
          mem_allocator_builder_->CreateBlock(tile_name, pre_output_name, 
              tile_output_name, outputTileShape, outputTilePos,
              output_bit_width/8, output_block_type);
        }
      }

      if (1)
      {
        for(int k = 0; k < output_tile_num; k++)
        {
          std::string tile_name = op_name + "_cast1.tile" + std::to_string(k);
          mem_allocator_builder_->CreateTile(tile_name, 0);
          std::string post_input_name = op_name + "_1";
          std::string post_output_name = output_tensor->GetName();
          op_output_name = post_input_name;
          printf("***********create post input tensor************\n");
          std::vector<int> inputTileShape;
          std::vector<int> inputTilePos;
          for (int i = 0; i < output_dim_num; i++)
          {
            int index = docTiling[(op_name).c_str()]["output_tile_pos"].Size() == 1 ? 0 : k;
            inputTilePos.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][index][i][0].GetInt());
            inputTileShape.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][index][i][1].GetInt() - inputTilePos[i]);
          }
          std::string tile_input_name = post_input_name;
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_input_name += "@";
              else
                  tile_input_name += ",";
              tile_input_name = tile_input_name + std::to_string(inputTilePos[i]);
          }
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_input_name += ":";
              else
                  tile_input_name += ",";
              tile_input_name = tile_input_name + std::to_string(inputTileShape[i]);
          }
          int input_bit_width = 16;
          int input_block_type = BLOCK_TYPE_INPUT;
          mem_allocator_builder_->CreateBlock(tile_name, post_input_name, 
              tile_input_name, inputTileShape, inputTilePos,
              input_bit_width/8, input_block_type);

          // create temp tensor
          std::string temp_name = tile_name + "@temp";
          std::vector<int> tempTileShape;
          std::vector<int> tempputTilePos;
          tempputTilePos.push_back(0);
          int input_len = 1;
          int output_len = 1;
          for (int i = 0; i < output_dim_num; i++)
          {
              input_len *= inputTileShape[i];
              output_len *= inputTileShape[i];
          }
          int align_size = 16*1024;
          int len = 0;
          if((input_len*2 + output_len) % align_size == 0)
          {
            len = (input_len*2 + output_len);
          }
          else
          {
            len = ((input_len*2 + output_len) / align_size + 1) * align_size;
          }
          tempTileShape.push_back(len);
          mem_allocator_builder_->CreateBlock(tile_name, temp_name, 
              temp_name, tempTileShape, tempputTilePos,
              1, BLOCK_TYPE_TEMP);

          // create output tensor
          printf("***********create post output tensor************\n");
          std::string tile_output_name = post_output_name;
          std::vector<int> outputTileShape;
          std::vector<int> outputTilePos;
          for (int i = 0; i < output_dim_num; i++)
          {
            int index = docTiling[(op_name).c_str()]["output_tile_pos"].Size() == 1 ? 0 : k;
            outputTilePos.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][index][i][0].GetInt());
            outputTileShape.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][index][i][1].GetInt() - inputTilePos[i]);
          }
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_output_name += "@";
              else
                  tile_output_name += ",";
              tile_output_name = tile_output_name + std::to_string(outputTilePos[i]);
          }
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_output_name += ":";
              else
                  tile_output_name += ",";
              tile_output_name = tile_output_name + std::to_string(outputTileShape[i]);
          }
          int output_bit_width = docAct[(post_output_name).c_str()]["bit_width"].GetInt();
          int output_block_type = BLOCK_TYPE_OUTPUT;
          if (layer_idx + 1 == docTiling["network"]["call_num"].GetInt())
          {
              output_block_type = BLOCK_TYPE_NET_OUTPUT;
          }
          mem_allocator_builder_->CreateBlock(tile_name, post_output_name, 
              tile_output_name, outputTileShape, outputTilePos,
              output_bit_width/8, output_block_type);
        }
      }
      
      for (int k = 0; k < output_tile_num; k++)
      {
          std::string tile_name = op_name + ".tile" + std::to_string(k);
          mem_allocator_builder_->CreateTile(tile_name, 0);

          // create input tensor
          printf("***********create input tensor************\n");
          std::vector<int> inputTileShape;
          std::vector<int> inputTilePos;
          for (int i = 0; i < output_dim_num; i++)
          {
              int index = docTiling[(op_name).c_str()]["input_tile_pos"].Size() == 1 ? 0 : k;
              inputTilePos.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][0].GetInt());
              inputTileShape.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][1].GetInt() - inputTilePos[i]);
          }
          std::string tile_input_name = op_input_name;
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_input_name += "@";
              else
                  tile_input_name += ",";
              tile_input_name = tile_input_name + std::to_string(inputTilePos[i]);
          }
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_input_name += ":";
              else
                  tile_input_name += ",";
              tile_input_name = tile_input_name + std::to_string(inputTileShape[i]);
          }
          int input_bit_width = 16;
          int input_block_type = BLOCK_TYPE_INPUT;
          mem_allocator_builder_->CreateBlock(tile_name, op_input_name, 
              tile_input_name, inputTileShape, inputTilePos,
              input_bit_width/8, input_block_type);
          // create output tensor
          printf("***********create output tensor************\n");
          std::string tile_output_name = op_output_name;
          std::vector<int> outputTileShape;
          std::vector<int> outputTilePos;
          for (int i = 0; i < output_dim_num; i++)
          {
              outputTilePos.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
              outputTileShape.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - outputTilePos[i]);
          }
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_output_name += "@";
              else
                  tile_output_name += ",";
              tile_output_name = tile_output_name + std::to_string(outputTilePos[i]);
          }
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_output_name += ":";
              else
                  tile_output_name += ",";
              tile_output_name = tile_output_name + std::to_string(outputTileShape[i]);
          }
          int output_bit_width = 16;
          int output_block_type = BLOCK_TYPE_OUTPUT;
          mem_allocator_builder_->CreateBlock(tile_name, op_output_name, 
              tile_output_name, outputTileShape, outputTilePos,
              output_bit_width/8, output_block_type);

      }
  }

  return outputs;
}

TensorVec VisVpuCodegen::Global_Pool2D_max(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";
  auto input_name = inputs.at(0);

  TensorPtr input_tensor = graph->FindTensor(input_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, GLOBAL_MAX_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetOutputTensor(0, output_tensor);
  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  if (doTiling)
  {
    std::string tiling_name;
    tiling_name = op_name;
    int output_tile_num = docTiling[(tiling_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(tiling_name).c_str()]["output_dim_num"].GetInt();

    int output_tile_dim = docTiling[(tiling_name).c_str()]["output_tile_dims"][0].GetInt();
    std::vector<TileTensorShape> tile_shape_list;
    std::vector<TileTensorShape> tile_shape_list_cast0;
    std::vector<TileTensorShape> tile_shape_list_cast1;
    for (int k = 0; k < output_tile_num; k ++)
    {
      std::vector<int> tileShape;
      std::vector<int> tilePos;
      std::vector<int> tileShapeInput;
      std::vector<int> tilePosInput;
      for (int i = 0; i < output_dim_num; ++i)
      {
        if (i == output_tile_dim)
        {
          tilePos.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
          tileShape.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - tilePos[i]);
          tilePosInput.push_back(docTiling[(tiling_name).c_str()]["input_tile_pos"][k][i][0].GetInt());
          tileShapeInput.push_back(docTiling[(tiling_name).c_str()]["input_tile_pos"][k][i][1].GetInt() - tilePos[i]);
        }
        else
        {
          assert(oshape[i] == (docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - tilePos[i]));
          tilePos.push_back(0);
          tileShape.push_back(oshape[i]);
          tilePosInput.push_back(0);
          tileShapeInput.push_back(docTiling[(tiling_name).c_str()]["input_tile_pos"][k][i][1].GetInt() - tilePos[i]);
        }
      }
      TileTensorShape tile_tensor_shape_cast0;
      tile_tensor_shape_cast0.shape = ShapeType(GetShapeType(tileShapeInput));
      tile_tensor_shape_cast0.pos = ShapeType(GetShapeType(tilePosInput));
      tile_shape_list_cast0.push_back(tile_tensor_shape_cast0);

      TileTensorShape tile_tensor_shape;
      tile_tensor_shape.shape = ShapeType(GetShapeType(tileShape));
      tile_tensor_shape.pos = ShapeType(GetShapeType(tilePos));
      tile_shape_list.push_back(tile_tensor_shape);
      
      TileTensorShape tile_tensor_shape_cast1;
      tile_tensor_shape_cast1.shape = ShapeType(GetShapeType(tileShape));
      tile_tensor_shape_cast1.pos = ShapeType(GetShapeType(tilePos));
      tile_shape_list_cast1.push_back(tile_tensor_shape_cast1);
    }
    if (useTiling)
    {
      RegisterTileSchema(op_name + "_cast0", tile_shape_list_cast0);
      RegisterTileSchema(op_name, tile_shape_list);
      RegisterTileSchema(op_name + "_cast1", tile_shape_list_cast1);
    }

    int group_index = docTiling[(tiling_name).c_str()]["group_index"].GetInt();
    int call_index = docTiling[(tiling_name).c_str()]["call_index"].GetInt();
    auto find = tile_sort_map.find(group_index);
    if (find != tile_sort_map.end()) 
    {
      size_t insert = 0;
      for (; insert < tile_sort_map[group_index]->call_index_list.size(); insert++)
      {
        if (tile_sort_map[group_index]->call_index_list[insert] > call_index)
        {
          break;
        }
      }
      tile_sort_map[group_index]->call_index_list.insert(tile_sort_map[group_index]->call_index_list.begin() + insert, call_index);
      tile_sort_map[group_index]->call_name_list.insert(tile_sort_map[group_index]->call_name_list.begin() + insert, op_name + "_cast1");

      tile_sort_map[group_index]->call_index_list.insert(tile_sort_map[group_index]->call_index_list.begin() + insert, call_index);
      tile_sort_map[group_index]->call_name_list.insert(tile_sort_map[group_index]->call_name_list.begin() + insert, op_name);

      tile_sort_map[group_index]->call_index_list.insert(tile_sort_map[group_index]->call_index_list.begin() + insert, call_index);
      tile_sort_map[group_index]->call_name_list.insert(tile_sort_map[group_index]->call_name_list.begin() + insert, op_name + "_cast0");
    }
    else
    {
      GroupInfoPtr group_info = GroupInfoPtr(new GroupInfo);
      group_info->index = group_index;
      group_info->tile_num = output_tile_num;

      group_info->call_index_list.push_back(call_index);
      group_info->call_name_list.push_back(op_name + "_cast0");

      group_info->call_index_list.push_back(call_index);
      group_info->call_name_list.push_back(op_name);

      group_info->call_index_list.push_back(call_index);
      group_info->call_name_list.push_back(op_name + "_cast1");

      tile_sort_map[group_index] = group_info;
    }
  }

  if (doAllocate)
  {
      printf("***********global ave doAllocate************\n");
      std::string op_input_name = input_tensor->GetName();
      std::string op_output_name = output_tensor->GetName();

      int output_tile_num = docTiling[(op_name).c_str()]["output_tile_num"].GetInt();
      int output_dim_num = docTiling[(op_name).c_str()]["output_dim_num"].GetInt();
      if (1)
      {
        for(int k = 0; k < output_tile_num; k++)
        {
          std::string tile_name = op_name + "_cast0.tile" + std::to_string(k);
          mem_allocator_builder_->CreateTile(tile_name, 0);
          std::string pre_input_name = input_tensor->GetName();
          std::string pre_output_name = op_name + "_cast0";
          op_input_name = pre_output_name;
          printf("***********create pre input tensor************\n");
          std::vector<int> inputTileShape;
          std::vector<int> inputTilePos;
          for (int i = 0; i < output_dim_num; i++)
          {
            int index = docTiling[(op_name).c_str()]["input_tile_pos"].Size() == 1 ? 0 : k;
            inputTilePos.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][0].GetInt());
            inputTileShape.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][1].GetInt() - inputTilePos[i]);
          }

          std::string tile_input_name = pre_input_name;
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_input_name += "@";
              else
                  tile_input_name += ",";
              tile_input_name = tile_input_name + std::to_string(inputTilePos[i]);
          }
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_input_name += ":";
              else
                  tile_input_name += ",";
              tile_input_name = tile_input_name + std::to_string(inputTileShape[i]);
          }
          int input_bit_width = 8;
          int input_block_type = BLOCK_TYPE_INPUT;
          mem_allocator_builder_->CreateBlock(tile_name, pre_input_name, 
              tile_input_name, inputTileShape, inputTilePos,
              input_bit_width/8, input_block_type);
          
          // std::cout << tile_input_name << " " << inputTileShape[2] << " " << inputTileShape[3] << std::endl;

          // create temp tensor
          std::string temp_name = tile_name + "@temp";
          std::vector<int> tempTileShape;
          std::vector<int> tempputTilePos;
          tempputTilePos.push_back(0);
          int input_len = 1;
          int output_len = 1;
          for (int i = 0; i < output_dim_num; i++)
          {
              input_len *= inputTileShape[i];
              output_len *= inputTileShape[i];
          }
          int align_size = 16*1024;
          int len = 0;
          if((input_len + output_len*2) % align_size == 0)
          {
            len = (input_len + output_len*2);
          }
          else
          {
            len = ((input_len + output_len*2) / align_size + 1) * align_size;
          }
          tempTileShape.push_back(len);
          mem_allocator_builder_->CreateBlock(tile_name, temp_name, 
              temp_name, tempTileShape, tempputTilePos,
              1, BLOCK_TYPE_TEMP);

          // create output tensor
          printf("***********create pre output tensor************\n");
          std::string tile_output_name = pre_output_name;
          std::vector<int> outputTileShape;
          std::vector<int> outputTilePos;
          for (int i = 0; i < output_dim_num; i++)
          {
            int index = docTiling[(op_name).c_str()]["input_tile_pos"].Size() == 1 ? 0 : k;
            outputTilePos.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][0].GetInt());
            outputTileShape.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][1].GetInt() - inputTilePos[i]);
          }
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_output_name += "@";
              else
                  tile_output_name += ",";
              tile_output_name = tile_output_name + std::to_string(outputTilePos[i]);
          }
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_output_name += ":";
              else
                  tile_output_name += ",";
              tile_output_name = tile_output_name + std::to_string(outputTileShape[i]);
          }
          int output_bit_width = 16;
          int output_block_type = BLOCK_TYPE_OUTPUT;
          
          mem_allocator_builder_->CreateBlock(tile_name, pre_output_name, 
              tile_output_name, outputTileShape, outputTilePos,
              output_bit_width/8, output_block_type);
        }
      }

      if (1)
      {
        for(int k = 0; k < output_tile_num; k++)
        {
          std::string tile_name = op_name + "_cast1.tile" + std::to_string(k);
          mem_allocator_builder_->CreateTile(tile_name, 0);
          std::string post_input_name = op_name + "_1";
          std::string post_output_name = output_tensor->GetName();
          op_output_name = post_input_name;
          printf("***********create post input tensor************\n");
          std::vector<int> inputTileShape;
          std::vector<int> inputTilePos;
          for (int i = 0; i < output_dim_num; i++)
          {
            int index = docTiling[(op_name).c_str()]["output_tile_pos"].Size() == 1 ? 0 : k;
            inputTilePos.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][index][i][0].GetInt());
            inputTileShape.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][index][i][1].GetInt() - inputTilePos[i]);
          }
          std::string tile_input_name = post_input_name;
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_input_name += "@";
              else
                  tile_input_name += ",";
              tile_input_name = tile_input_name + std::to_string(inputTilePos[i]);
          }
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_input_name += ":";
              else
                  tile_input_name += ",";
              tile_input_name = tile_input_name + std::to_string(inputTileShape[i]);
          }
          int input_bit_width = 16;
          int input_block_type = BLOCK_TYPE_INPUT;
          mem_allocator_builder_->CreateBlock(tile_name, post_input_name, 
              tile_input_name, inputTileShape, inputTilePos,
              input_bit_width/8, input_block_type);

          // create temp tensor
          std::string temp_name = tile_name + "@temp";
          std::vector<int> tempTileShape;
          std::vector<int> tempputTilePos;
          tempputTilePos.push_back(0);
          int input_len = 1;
          int output_len = 1;
          for (int i = 0; i < output_dim_num; i++)
          {
              input_len *= inputTileShape[i];
              output_len *= inputTileShape[i];
          }
          int align_size = 16*1024;
          int len = 0;
          if((input_len*2 + output_len) % align_size == 0)
          {
            len = (input_len*2 + output_len);
          }
          else
          {
            len = ((input_len*2 + output_len) / align_size + 1) * align_size;
          }
          tempTileShape.push_back(len);
          mem_allocator_builder_->CreateBlock(tile_name, temp_name, 
              temp_name, tempTileShape, tempputTilePos,
              1, BLOCK_TYPE_TEMP);

          // create output tensor
          printf("***********create post output tensor************\n");
          std::string tile_output_name = post_output_name;
          std::vector<int> outputTileShape;
          std::vector<int> outputTilePos;
          for (int i = 0; i < output_dim_num; i++)
          {
            int index = docTiling[(op_name).c_str()]["output_tile_pos"].Size() == 1 ? 0 : k;
            outputTilePos.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][index][i][0].GetInt());
            outputTileShape.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][index][i][1].GetInt() - inputTilePos[i]);
          }
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_output_name += "@";
              else
                  tile_output_name += ",";
              tile_output_name = tile_output_name + std::to_string(outputTilePos[i]);
          }
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_output_name += ":";
              else
                  tile_output_name += ",";
              tile_output_name = tile_output_name + std::to_string(outputTileShape[i]);
          }
          int output_bit_width = docAct[(post_output_name).c_str()]["bit_width"].GetInt();
          int output_block_type = BLOCK_TYPE_OUTPUT;
          if (layer_idx + 1 == docTiling["network"]["call_num"].GetInt())
          {
              output_block_type = BLOCK_TYPE_NET_OUTPUT;
          }
          mem_allocator_builder_->CreateBlock(tile_name, post_output_name, 
              tile_output_name, outputTileShape, outputTilePos,
              output_bit_width/8, output_block_type);
        }
      }
      
      for (int k = 0; k < output_tile_num; k++)
      {
          std::string tile_name = op_name + ".tile" + std::to_string(k);
          mem_allocator_builder_->CreateTile(tile_name, 0);

          // create input tensor
          printf("***********create input tensor************\n");
          std::vector<int> inputTileShape;
          std::vector<int> inputTilePos;
          for (int i = 0; i < output_dim_num; i++)
          {
              int index = docTiling[(op_name).c_str()]["input_tile_pos"].Size() == 1 ? 0 : k;
              inputTilePos.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][0].GetInt());
              inputTileShape.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][1].GetInt() - inputTilePos[i]);
          }
          std::string tile_input_name = op_input_name;
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_input_name += "@";
              else
                  tile_input_name += ",";
              tile_input_name = tile_input_name + std::to_string(inputTilePos[i]);
          }
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_input_name += ":";
              else
                  tile_input_name += ",";
              tile_input_name = tile_input_name + std::to_string(inputTileShape[i]);
          }
          int input_bit_width = 16;
          int input_block_type = BLOCK_TYPE_INPUT;
          mem_allocator_builder_->CreateBlock(tile_name, op_input_name, 
              tile_input_name, inputTileShape, inputTilePos,
              input_bit_width/8, input_block_type);
          // create output tensor
          printf("***********create output tensor************\n");
          std::string tile_output_name = op_output_name;
          std::vector<int> outputTileShape;
          std::vector<int> outputTilePos;
          for (int i = 0; i < output_dim_num; i++)
          {
              outputTilePos.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
              outputTileShape.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - outputTilePos[i]);
          }
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_output_name += "@";
              else
                  tile_output_name += ",";
              tile_output_name = tile_output_name + std::to_string(outputTilePos[i]);
          }
          for (int i = 0; i < output_dim_num; i++)
          {
              if (i==0)
                  tile_output_name += ":";
              else
                  tile_output_name += ",";
              tile_output_name = tile_output_name + std::to_string(outputTileShape[i]);
          }
          int output_bit_width = 16;
          int output_block_type = BLOCK_TYPE_OUTPUT;
          mem_allocator_builder_->CreateBlock(tile_name, op_output_name, 
              tile_output_name, outputTileShape, outputTilePos,
              output_bit_width/8, output_block_type);

      }
  }

  return outputs;
}

TensorVec VisVpuCodegen::LeakyReLU(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";
  auto input_name = inputs.at(0);

  TensorPtr input_tensor = graph->FindTensor(input_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, LEAKYRELU_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetOutputTensor(0, output_tensor);
  // parse attr
  const auto* attrs = call->attrs.as<LeakyReluAttrs>();  //

  CHECK(attrs) << "not LeakyReluAttrs node";  //

  float Alpha_float = static_cast<float>(attrs->alpha);

  // config param
  LeakyreluOpPtr op_ptr = std::dynamic_pointer_cast<LeakyreluOp>(node->GetOp());  //

  op_ptr->SetAlpha(Alpha_float);
  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  if (doTiling)
  {
    std::string tiling_name;
    tiling_name = op_name;
    int output_tile_num = docTiling[(tiling_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(tiling_name).c_str()]["output_dim_num"].GetInt();

    int output_tile_dim = docTiling[(tiling_name).c_str()]["output_tile_dims"][0].GetInt();
    std::vector<TileTensorShape> tile_shape_list;
    for (int k = 0; k < output_tile_num; k ++)
    {
      std::vector<int> tileShape;
      std::vector<int> tilePos;
      for (int i = 0; i < output_dim_num; ++i)
      {
        if (i == output_tile_dim)
        {
          tilePos.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
          tileShape.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - tilePos[i]);
        }
        else
        {
          assert(oshape[i] == docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt());
          tilePos.push_back(0);
          tileShape.push_back(oshape[i]);
        }
      }
      TileTensorShape tile_tensor_shape;
      tile_tensor_shape.shape = ShapeType(GetShapeType(tileShape));
      tile_tensor_shape.pos = ShapeType(GetShapeType(tilePos));
      tile_shape_list.push_back(tile_tensor_shape);
    }
    if (useTiling)
      RegisterTileSchema(op_name, tile_shape_list);

    int group_index = docTiling[(tiling_name).c_str()]["group_index"].GetInt();
    int call_index = docTiling[(tiling_name).c_str()]["call_index"].GetInt();
    auto find = tile_sort_map.find(group_index);
    if (find != tile_sort_map.end()) 
    {
      size_t insert = 0;
      for (; insert < tile_sort_map[group_index]->call_index_list.size(); insert++)
      {
        if (tile_sort_map[group_index]->call_index_list[insert] > call_index)
        {
          break;
        }
      }

      tile_sort_map[group_index]->call_index_list.insert(tile_sort_map[group_index]->call_index_list.begin() + insert, call_index);
      tile_sort_map[group_index]->call_name_list.insert(tile_sort_map[group_index]->call_name_list.begin() + insert, op_name);
    }
    else
    {
      GroupInfoPtr group_info = GroupInfoPtr(new GroupInfo);
      group_info->index = group_index;
      group_info->tile_num = output_tile_num;
      group_info->call_index_list.push_back(call_index);
      group_info->call_name_list.push_back(op_name);
      tile_sort_map[group_index] = group_info;
    }
  }

  if (doAllocate)
  {
    printf("***********doAllocate************\n");
    int output_tile_num = docTiling[(op_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(op_name).c_str()]["output_dim_num"].GetInt();
    // int output_tile_dim = docTiling[(op_name).c_str()]["output_tile_dims"][0].GetInt();
    std::string op_input_name = input_tensor->GetName();
    std::string op_output_name = output_tensor->GetName();

    for (int k = 0; k < output_tile_num; k++)
    {
      std::string tile_name = op_name + ".tile" + std::to_string(k);
      mem_allocator_builder_->CreateTile(tile_name, 0);

      // create input tensor
      printf("***********create input tensor************\n");
      std::string tile_input_name = op_input_name;
      std::vector<int> inputTileShape;
      std::vector<int> inputTilePos;
      for (int i = 0; i < output_dim_num; i++)
      {
        int index = docTiling[(op_name).c_str()]["input_tile_pos"].Size() == 1 ? 0 : k;
        inputTilePos.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][0].GetInt());
        inputTileShape.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][1].GetInt() - inputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_input_name += "@";
        else
          tile_input_name += ",";
        tile_input_name = tile_input_name + std::to_string(inputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_input_name += ":";
        else
          tile_input_name += ",";
        tile_input_name = tile_input_name + std::to_string(inputTileShape[i]);
      }
      int input_bit_width = docAct[(op_input_name).c_str()]["bit_width"].GetInt();
      int input_block_type = BLOCK_TYPE_INPUT;
      mem_allocator_builder_->CreateBlock(tile_name, op_input_name, 
          tile_input_name, inputTileShape, inputTilePos,
          input_bit_width/8, input_block_type);

      // create output tensor
      printf("***********create output tensor************\n");
      std::string tile_output_name = op_output_name;
      std::vector<int> outputTileShape;
      std::vector<int> outputTilePos;
      for (int i = 0; i < output_dim_num; i++)
      {
        outputTilePos.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
        outputTileShape.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - outputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_output_name += "@";
        else
          tile_output_name += ",";
        tile_output_name = tile_output_name + std::to_string(outputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_output_name += ":";
        else
          tile_output_name += ",";
        tile_output_name = tile_output_name + std::to_string(outputTileShape[i]);
      }
      int output_bit_width = docAct[(op_output_name).c_str()]["bit_width"].GetInt();
      int output_block_type = BLOCK_TYPE_OUTPUT;
      if (layer_idx + 1 == docTiling["network"]["call_num"].GetInt())
      {
        output_block_type = BLOCK_TYPE_NET_OUTPUT;
      }
      mem_allocator_builder_->CreateBlock(tile_name, op_output_name, 
          tile_output_name, outputTileShape, outputTilePos,
          output_bit_width/8, output_block_type);
      mem_allocator_builder_->CreateReuse(tile_name, tile_input_name, tile_output_name, false);

    }
  }

  return outputs;
}

TensorVec VisVpuCodegen::Pool2D_max(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";
  auto input_name = inputs.at(0);

  TensorPtr input_tensor = graph->FindTensor(input_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, MAX_POOL_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetOutputTensor(0, output_tensor);

  // parse attr
  const auto* attrs = call->attrs.as<MaxPool2DAttrs>();  //

  CHECK(attrs) << "not MaxPool2DAttrs node";  //

  auto strides = GetConcrete<int>(attrs->strides);
  auto padding = GetConcrete<int>(attrs->padding);
  auto pool_size = GetConcrete<int>(attrs->pool_size);

  // config param
  MaxPoolOpPtr op_ptr = std::dynamic_pointer_cast<MaxPoolOp>(node->GetOp());  //

  PoolParam& param = op_ptr->GetParam();

  param.pad_w0 = padding[0];
  param.pad_h0 = padding[1];
  param.pad_w1 = padding[2];
  param.pad_h1 = padding[3];
  param.kernel_h = pool_size[0];
  param.kernel_w = pool_size[1];
  param.stride_h = strides[0];
  param.stride_w = strides[1];

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  if (doTiling)
  {
    std::string tiling_name;
    tiling_name = op_name;
    int output_tile_num = docTiling[(tiling_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(tiling_name).c_str()]["output_dim_num"].GetInt();
    assert(output_dim_num == 4);
    int output_tile_dim = docTiling[(tiling_name).c_str()]["output_tile_dims"][0].GetInt();
    std::vector<TileTensorShape> tile_shape_list;
    for (int k = 0; k < output_tile_num; k ++)
    {
      std::vector<int> tileShape;
      std::vector<int> tilePos;
      for (int i = 0; i < output_dim_num; ++i)
      {
        if (i == output_tile_dim)
        {
          tilePos.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
          tileShape.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - tilePos[i]);
        }
        else
        {
          assert(oshape[i] == docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt());
          tilePos.push_back(0);
          tileShape.push_back(oshape[i]);
        }
      }
      TileTensorShape tile_tensor_shape;
      tile_tensor_shape.shape = ShapeType(GetShapeType(tileShape));
      tile_tensor_shape.pos = ShapeType(GetShapeType(tilePos));
      tile_shape_list.push_back(tile_tensor_shape);
    }
    if (useTiling)
      RegisterTileSchema(op_name, tile_shape_list);

    int group_index = docTiling[(tiling_name).c_str()]["group_index"].GetInt();
    int call_index = docTiling[(tiling_name).c_str()]["call_index"].GetInt();
    auto find = tile_sort_map.find(group_index);
    if (find != tile_sort_map.end()) 
    {
      size_t insert = 0;
      for (; insert < tile_sort_map[group_index]->call_index_list.size(); insert++)
      {
        if (tile_sort_map[group_index]->call_index_list[insert] > call_index)
        {
          break;
        }
      }

      tile_sort_map[group_index]->call_index_list.insert(tile_sort_map[group_index]->call_index_list.begin() + insert, call_index);
      tile_sort_map[group_index]->call_name_list.insert(tile_sort_map[group_index]->call_name_list.begin() + insert, op_name);
    }
    else
    {
      GroupInfoPtr group_info = GroupInfoPtr(new GroupInfo);
      group_info->index = group_index;
      group_info->tile_num = output_tile_num;
      group_info->call_index_list.push_back(call_index);
      group_info->call_name_list.push_back(op_name);
      tile_sort_map[group_index] = group_info;
    }
  }

  if (doAllocate)
  {
    printf("***********doAllocate************\n");
    int output_tile_num = docTiling[(op_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(op_name).c_str()]["output_dim_num"].GetInt();
    // int output_tile_dim = docTiling[(op_name).c_str()]["output_tile_dims"][0].GetInt();
    std::string op_input_name = input_tensor->GetName();
    std::string op_output_name = output_tensor->GetName();

    for (int k = 0; k < output_tile_num; k++)
    {
      std::string tile_name = op_name + ".tile" + std::to_string(k);
      mem_allocator_builder_->CreateTile(tile_name, 0);

      // create input tensor
      printf("***********create input tensor************\n");
      std::string tile_input_name = op_input_name;
      std::vector<int> inputTileShape;
      std::vector<int> inputTilePos;
      for (int i = 0; i < output_dim_num; i++)
      {
        int index = docTiling[(op_name).c_str()]["input_tile_pos"].Size() == 1 ? 0 : k;
        inputTilePos.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][0].GetInt());
        inputTileShape.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][1].GetInt() - inputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_input_name += "@";
        else
          tile_input_name += ",";
        tile_input_name = tile_input_name + std::to_string(inputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_input_name += ":";
        else
          tile_input_name += ",";
        tile_input_name = tile_input_name + std::to_string(inputTileShape[i]);
      }
      int input_bit_width = docAct[(op_input_name).c_str()]["bit_width"].GetInt();
      int input_block_type = BLOCK_TYPE_INPUT;
      mem_allocator_builder_->CreateBlock(tile_name, op_input_name, 
          tile_input_name, inputTileShape, inputTilePos,
          input_bit_width/8, input_block_type);

      // create output tensor
      printf("***********create output tensor************\n");
      std::string tile_output_name = op_output_name;
      std::vector<int> outputTileShape;
      std::vector<int> outputTilePos;
      for (int i = 0; i < output_dim_num; i++)
      {
        outputTilePos.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
        outputTileShape.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - outputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_output_name += "@";
        else
          tile_output_name += ",";
        tile_output_name = tile_output_name + std::to_string(outputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_output_name += ":";
        else
          tile_output_name += ",";
        tile_output_name = tile_output_name + std::to_string(outputTileShape[i]);
      }
      int output_bit_width = docAct[(op_output_name).c_str()]["bit_width"].GetInt();
      int output_block_type = BLOCK_TYPE_OUTPUT;
      if (layer_idx + 1 == docTiling["network"]["call_num"].GetInt())
      {
        output_block_type = BLOCK_TYPE_NET_OUTPUT;
      }
      mem_allocator_builder_->CreateBlock(tile_name, op_output_name, 
          tile_output_name, outputTileShape, outputTilePos,
          output_bit_width/8, output_block_type);

    }
  }

  return outputs;
}

TensorVec VisVpuCodegen::Mul(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 2) << "number of inputs need to corrected";
  auto input_name = inputs.at(0);
  auto kernel_name = inputs.at(1);

  TensorPtr input_tensor = graph->FindTensor(input_name);
  TensorPtr kernel_tensor = graph->FindTensor(kernel_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, (input_name == kernel_name) ? SQUARE_OP : MUL_OP);  //

  if (input_name == kernel_name) {
    node->SetInputTensor(0, input_tensor);
  } else {
    node->SetInputTensor(0, input_tensor);
    node->SetInputTensor(1, kernel_tensor);
  }

  node->SetOutputTensor(0, output_tensor);

  auto kernel_shape = kernel_tensor->GetShape();
  auto input_shape = input_tensor->GetShape();

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};
  if (input_tensor->GetTensorType() == kTT_Const) {
    if (input_shape.Volume() != 1 && input_shape.dim_num == 1 &&
        input_shape.dim[input_shape.dim_num - 1] == kernel_shape.dim[kernel_shape.dim_num - 1]) {
      input_shape.dim_num = kernel_shape.dim_num;
      for (int i = 0; i < kernel_shape.dim_num; ++i) input_shape.dim[i] = 1;
      input_shape.dim[kernel_shape.dim_num - 1] = kernel_shape.dim[kernel_shape.dim_num - 1];
      input_tensor->SetShape(input_shape);
    }
    auto op_node = call->op.as<OpNode>();
    auto operator_id = op_node->name;
    std::string kernel_quant_name;
    kernel_quant_name += operator_id + "_weight_" + std::to_string(layer_idx) + "_0:in";
    input_tensor->SetName(kernel_quant_name);
    std::vector<float> channel_scale(input_shape.dim[0]);
    std::vector<int> channel_zero_point(input_shape.dim[0]);
    std::vector<float> channel_max(input_shape.dim[0]);
    std::vector<float> channel_min(input_shape.dim[0]);

    const rapidjson::Value& scale = docWgt[(kernel_quant_name).c_str()]["scale"];
    const rapidjson::Value& zero_point = docWgt[(kernel_quant_name).c_str()]["zero_point"];
    const rapidjson::Value& max = docWgt[(kernel_quant_name).c_str()]["max_value"];
    const rapidjson::Value& min = docWgt[(kernel_quant_name).c_str()]["min_value"];
    if (scale.Size() > 1) {
      assert(scale.Size() == input_shape.dim[0]);
      for (int i = 0; i < input_shape.dim[0]; i++) {
        channel_scale[i] = scale[i].GetFloat();
        if (zero_point.IsArray()) {
          channel_zero_point[i] = zero_point[i].GetFloat();
        } else {
          channel_zero_point[i] = zero_point.GetFloat();
        }
        channel_max[i] = max[i].GetFloat();
        channel_min[i] = min[i].GetFloat();
      }

      input_tensor->SetChannelScale(channel_scale.data(), input_shape.dim[0]);
      input_tensor->SetChannelZeroPoint(channel_zero_point.data(), input_shape.dim[0]);
      input_tensor->SetChannelMax(channel_max.data(), input_shape.dim[0]);
      input_tensor->SetChannelMin(channel_min.data(), input_shape.dim[0]);
      if (std::strcmp(docWgt[(kernel_quant_name).c_str()]["quantizer"].GetString(), "Symmetric") ==
          0)
        input_tensor->SetQuantType(kQT_Sym);
      else
        input_tensor->SetQuantType(kQT_Asym);
    } else {
      input_tensor->SetScale(docWgt[(kernel_quant_name).c_str()]["scale"][0].GetFloat());

      input_tensor->SetMax(docWgt[(kernel_quant_name).c_str()]["max_value"][0].GetFloat());
      input_tensor->SetMin(docWgt[(kernel_quant_name).c_str()]["min_value"][0].GetFloat());
      if (std::strcmp(docWgt[(kernel_quant_name).c_str()]["quantizer"].GetString(), "Symmetric") ==
          0) {
        input_tensor->SetQuantType(kQT_Sym);
        input_tensor->SetZeroPoint(docWgt[(kernel_quant_name).c_str()]["zero_point"].GetFloat());
      } else {
        input_tensor->SetQuantType(kQT_Asym);
        input_tensor->SetZeroPoint(docWgt[(kernel_quant_name).c_str()]["zero_point"][0].GetFloat());
      }
    }
  }
  // quant weight
  if (kernel_tensor->GetTensorType() == kTT_Const) {
    if (kernel_shape.Volume() != 1 && kernel_shape.dim_num == 1 &&
        kernel_shape.dim[kernel_shape.dim_num - 1] == input_shape.dim[input_shape.dim_num - 1]) {
      kernel_shape.dim_num = input_shape.dim_num;
      for (int i = 0; i < input_shape.dim_num; ++i) kernel_shape.dim[i] = 1;
      kernel_shape.dim[input_shape.dim_num - 1] = input_shape.dim[input_shape.dim_num - 1];
      kernel_tensor->SetShape(kernel_shape);
    }
    auto op_node = call->op.as<OpNode>();
    auto operator_id = op_node->name;
    std::string kernel_quant_name;
    kernel_quant_name += operator_id + "_weight_" + std::to_string(layer_idx) + "_0:in";
    kernel_tensor->SetName(kernel_quant_name);

    // if some Muls have the same weight data but need different shape, create a new weight.
    // for example, mul weight is 256,  need 1*256 and 1*1*256.
    if (input_shape.Volume() != kernel_shape.Volume()) {
      TensorPtr kernel_tensor_new =
          graph->CreateTensor(kernel_quant_name, kTT_Const, kernel_tensor->GetDataType());
      auto p = kernel_tensor->GetData();
      kernel_tensor_new->SetShape(kernel_tensor->GetShape());
      kernel_tensor_new->SetData(p, kernel_tensor->GetMemSize());

      node->SetInputTensor(1, kernel_tensor_new);

      return outputs;
    }
    std::vector<float> channel_scale(kernel_shape.dim[0]);
    std::vector<int> channel_zero_point(kernel_shape.dim[0]);
    std::vector<float> channel_max(kernel_shape.dim[0]);
    std::vector<float> channel_min(kernel_shape.dim[0]);

    const rapidjson::Value& scale = docWgt[(kernel_quant_name).c_str()]["scale"];
    const rapidjson::Value& zero_point = docWgt[(kernel_quant_name).c_str()]["zero_point"];
    const rapidjson::Value& max = docWgt[(kernel_quant_name).c_str()]["max_value"];
    const rapidjson::Value& min = docWgt[(kernel_quant_name).c_str()]["min_value"];
    if (scale.Size() > 1) {
      assert(scale.Size() == kernel_shape.dim[0]);
      for (int i = 0; i < kernel_shape.dim[0]; i++) {
        channel_scale[i] = scale[i].GetFloat();
        if (zero_point.IsArray()) {
          channel_zero_point[i] = zero_point[i].GetFloat();
        } else {
          channel_zero_point[i] = zero_point.GetFloat();
        }
        channel_max[i] = max[i].GetFloat();
        channel_min[i] = min[i].GetFloat();
      }

      kernel_tensor->SetChannelScale(channel_scale.data(), kernel_shape.dim[0]);
      kernel_tensor->SetChannelZeroPoint(channel_zero_point.data(), kernel_shape.dim[0]);
      kernel_tensor->SetChannelMax(channel_max.data(), kernel_shape.dim[0]);
      kernel_tensor->SetChannelMin(channel_min.data(), kernel_shape.dim[0]);
      if (std::strcmp(docWgt[(kernel_quant_name).c_str()]["quantizer"].GetString(), "Symmetric") ==
          0)
        kernel_tensor->SetQuantType(kQT_Sym);
      else
        kernel_tensor->SetQuantType(kQT_Asym);
    } else {
      kernel_tensor->SetScale(docWgt[(kernel_quant_name).c_str()]["scale"][0].GetFloat());

      kernel_tensor->SetMax(docWgt[(kernel_quant_name).c_str()]["max_value"][0].GetFloat());
      kernel_tensor->SetMin(docWgt[(kernel_quant_name).c_str()]["min_value"][0].GetFloat());
      if (std::strcmp(docWgt[(kernel_quant_name).c_str()]["quantizer"].GetString(), "Symmetric") ==
          0) {
        kernel_tensor->SetQuantType(kQT_Sym);
        kernel_tensor->SetZeroPoint(docWgt[(kernel_quant_name).c_str()]["zero_point"].GetFloat());
      } else {
        kernel_tensor->SetQuantType(kQT_Asym);
        kernel_tensor->SetZeroPoint(
            docWgt[(kernel_quant_name).c_str()]["zero_point"][0].GetFloat());
      }
    }
  }

  if (doTiling)
  {
    std::string tiling_name;
    tiling_name = op_name;
    int output_tile_num = docTiling[(tiling_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(tiling_name).c_str()]["output_dim_num"].GetInt();

    int output_tile_dim = docTiling[(tiling_name).c_str()]["output_tile_dims"][0].GetInt();
    std::vector<TileTensorShape> tile_shape_list;
    for (int k = 0; k < output_tile_num; k ++)
    {
      std::vector<int> tileShape;
      std::vector<int> tilePos;
      for (int i = 0; i < output_dim_num; ++i)
      {
        if (i == output_tile_dim)
        {
          tilePos.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
          tileShape.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - tilePos[i]);
        }
        else
        {
          assert(oshape[i] == docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt());
          tilePos.push_back(0);
          tileShape.push_back(oshape[i]);
        }
      }
      TileTensorShape tile_tensor_shape;
      tile_tensor_shape.shape = ShapeType(GetShapeType(tileShape));
      tile_tensor_shape.pos = ShapeType(GetShapeType(tilePos));
      tile_shape_list.push_back(tile_tensor_shape);
    }
    if (useTiling)
      RegisterTileSchema(op_name, tile_shape_list);

    int group_index = docTiling[(tiling_name).c_str()]["group_index"].GetInt();
    int call_index = docTiling[(tiling_name).c_str()]["call_index"].GetInt();
    auto find = tile_sort_map.find(group_index);
    if (find != tile_sort_map.end()) 
    {
      size_t insert = 0;
      for (; insert < tile_sort_map[group_index]->call_index_list.size(); insert++)
      {
        if (tile_sort_map[group_index]->call_index_list[insert] > call_index)
        {
          break;
        }
      }

      tile_sort_map[group_index]->call_index_list.insert(tile_sort_map[group_index]->call_index_list.begin() + insert, call_index);
      tile_sort_map[group_index]->call_name_list.insert(tile_sort_map[group_index]->call_name_list.begin() + insert, op_name);
    }
    else
    {
      GroupInfoPtr group_info = GroupInfoPtr(new GroupInfo);
      group_info->index = group_index;
      group_info->tile_num = output_tile_num;
      group_info->call_index_list.push_back(call_index);
      group_info->call_name_list.push_back(op_name);
      tile_sort_map[group_index] = group_info;
    }
  }

  return outputs;
}

TensorVec VisVpuCodegen::Exp(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";
  auto input_name = inputs.at(0);

  TensorPtr input_tensor = graph->FindTensor(input_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, EXP_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetOutputTensor(0, output_tensor);

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  return outputs;
}

TensorVec VisVpuCodegen::Tanh(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";
  auto input_name = inputs.at(0);

  TensorPtr input_tensor = graph->FindTensor(input_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, TANH_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetOutputTensor(0, output_tensor);

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  return outputs;
}

TensorVec VisVpuCodegen::Sqrt(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";
  auto input_name = inputs.at(0);

  TensorPtr input_tensor = graph->FindTensor(input_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, SQRT_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetOutputTensor(0, output_tensor);

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  return outputs;
}

TensorVec VisVpuCodegen::Softmax(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";
  auto input_name = inputs.at(0);

  TensorPtr input_tensor = graph->FindTensor(input_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, SOFTMAX_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetOutputTensor(0, output_tensor);

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  return outputs;
}

TensorVec VisVpuCodegen::Unsqueeze(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";
  auto input_name = inputs.at(0);

  TensorPtr input_tensor = graph->FindTensor(input_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, RESHAPE_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetOutputTensor(0, output_tensor);

  // config param
  ReshapeOpPtr op_ptr = std::dynamic_pointer_cast<ReshapeOp>(node->GetOp());  //

  op_ptr->SetTargetShape(output_tensor->GetShape());

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  return outputs;
}

TensorVec VisVpuCodegen::Matmul(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 2) << "number of inputs need to corrected";
  auto input_name = inputs.at(0);

  TensorPtr input_tensor = graph->FindTensor(input_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, MATMUL_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetOutputTensor(0, output_tensor);

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  return outputs;
}

TensorVec VisVpuCodegen::Bmatmul(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 2) << "number of inputs need to corrected";
  auto input_name = inputs.at(0);
  auto kernel_name = inputs.at(1);

  TensorPtr input_tensor = graph->FindTensor(input_name);
  TensorPtr kernel_tensor = graph->FindTensor(kernel_name);
  auto kernel_shape = kernel_tensor->GetShape();
  auto input_shape = input_tensor->GetShape();

  // force to change N and K for test comile.
#if 0  
  int K = kernel_shape.dim[kernel_shape.dim_num-1];
  int N = kernel_shape.dim[kernel_shape.dim_num-2];

  kernel_shape.dim[kernel_shape.dim_num-1] = N;
  kernel_shape.dim[kernel_shape.dim_num-2] = K;
#endif

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, BMATMUL_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetInputTensor(1, kernel_tensor);
  node->SetOutputTensor(0, output_tensor);

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  if (input_tensor->GetTensorType() == kTT_Const) {
    auto op_node = call->op.as<OpNode>();
    auto operator_id = op_node->name;
    std::string kernel_quant_name;
    kernel_quant_name += operator_id + "_weight_" + std::to_string(layer_idx) + "_0:in";
    input_tensor->SetName(kernel_quant_name);
    std::vector<float> channel_scale(input_shape.dim[0]);
    std::vector<int> channel_zero_point(input_shape.dim[0]);
    std::vector<float> channel_max(input_shape.dim[0]);
    std::vector<float> channel_min(input_shape.dim[0]);

    const rapidjson::Value& scale = docWgt[(kernel_quant_name).c_str()]["scale"];
    const rapidjson::Value& zero_point = docWgt[(kernel_quant_name).c_str()]["zero_point"];
    const rapidjson::Value& max = docWgt[(kernel_quant_name).c_str()]["max_value"];
    const rapidjson::Value& min = docWgt[(kernel_quant_name).c_str()]["min_value"];
    if (scale.Size() > 1) {
      assert(scale.Size() == input_shape.dim[0]);
      for (int i = 0; i < input_shape.dim[0]; i++) {
        channel_scale[i] = scale[i].GetFloat();
        if (zero_point.IsArray()) {
          channel_zero_point[i] = zero_point[i].GetFloat();
        } else {
          channel_zero_point[i] = zero_point.GetFloat();
        }
        channel_max[i] = max[i].GetFloat();
        channel_min[i] = min[i].GetFloat();
      }

      input_tensor->SetChannelScale(channel_scale.data(), input_shape.dim[0]);
      input_tensor->SetChannelZeroPoint(channel_zero_point.data(), input_shape.dim[0]);
      input_tensor->SetChannelMax(channel_max.data(), input_shape.dim[0]);
      input_tensor->SetChannelMin(channel_min.data(), input_shape.dim[0]);
      if (std::strcmp(docWgt[(kernel_quant_name).c_str()]["quantizer"].GetString(), "Symmetric") ==
          0)
        input_tensor->SetQuantType(kQT_Sym);
      else
        input_tensor->SetQuantType(kQT_Asym);
    } else {
      input_tensor->SetScale(docWgt[(kernel_quant_name).c_str()]["scale"][0].GetFloat());

      input_tensor->SetMax(docWgt[(kernel_quant_name).c_str()]["max_value"][0].GetFloat());
      input_tensor->SetMin(docWgt[(kernel_quant_name).c_str()]["min_value"][0].GetFloat());
      if (std::strcmp(docWgt[(kernel_quant_name).c_str()]["quantizer"].GetString(), "Symmetric") ==
          0) {
        input_tensor->SetQuantType(kQT_Sym);
        input_tensor->SetZeroPoint(docWgt[(kernel_quant_name).c_str()]["zero_point"].GetFloat());
      } else {
        input_tensor->SetQuantType(kQT_Asym);
        input_tensor->SetZeroPoint(docWgt[(kernel_quant_name).c_str()]["zero_point"][0].GetFloat());
      }
    }
  }
  // quant weight
  if (kernel_tensor->GetTensorType() == kTT_Const) {
    auto op_node = call->op.as<OpNode>();
    auto operator_id = op_node->name;
    std::string kernel_quant_name;
    kernel_quant_name += operator_id + "_weight_" + std::to_string(layer_idx) + "_0:in";
    kernel_tensor->SetName(kernel_quant_name);
    std::vector<float> channel_scale(kernel_shape.dim[0]);
    std::vector<int> channel_zero_point(kernel_shape.dim[0]);
    std::vector<float> channel_max(kernel_shape.dim[0]);
    std::vector<float> channel_min(kernel_shape.dim[0]);

    const rapidjson::Value& scale = docWgt[(kernel_quant_name).c_str()]["scale"];
    const rapidjson::Value& zero_point = docWgt[(kernel_quant_name).c_str()]["zero_point"];
    const rapidjson::Value& max = docWgt[(kernel_quant_name).c_str()]["max_value"];
    const rapidjson::Value& min = docWgt[(kernel_quant_name).c_str()]["min_value"];
    if (scale.Size() > 1) {
      assert(scale.Size() == kernel_shape.dim[0]);
      for (int i = 0; i < kernel_shape.dim[0]; i++) {
        channel_scale[i] = scale[i].GetFloat();
        if (zero_point.IsArray()) {
          channel_zero_point[i] = zero_point[i].GetFloat();
        } else {
          channel_zero_point[i] = zero_point.GetFloat();
        }
        channel_max[i] = max[i].GetFloat();
        channel_min[i] = min[i].GetFloat();
      }

      kernel_tensor->SetChannelScale(channel_scale.data(), kernel_shape.dim[0]);
      kernel_tensor->SetChannelZeroPoint(channel_zero_point.data(), kernel_shape.dim[0]);
      kernel_tensor->SetChannelMax(channel_max.data(), kernel_shape.dim[0]);
      kernel_tensor->SetChannelMin(channel_min.data(), kernel_shape.dim[0]);
      if (std::strcmp(docWgt[(kernel_quant_name).c_str()]["quantizer"].GetString(), "Symmetric") ==
          0)
        kernel_tensor->SetQuantType(kQT_Sym);
      else
        kernel_tensor->SetQuantType(kQT_Asym);
    } else {
      kernel_tensor->SetScale(docWgt[(kernel_quant_name).c_str()]["scale"][0].GetFloat());

      kernel_tensor->SetMax(docWgt[(kernel_quant_name).c_str()]["max_value"][0].GetFloat());
      kernel_tensor->SetMin(docWgt[(kernel_quant_name).c_str()]["min_value"][0].GetFloat());
      if (std::strcmp(docWgt[(kernel_quant_name).c_str()]["quantizer"].GetString(), "Symmetric") ==
          0) {
        kernel_tensor->SetQuantType(kQT_Sym);
        kernel_tensor->SetZeroPoint(docWgt[(kernel_quant_name).c_str()]["zero_point"].GetFloat());
      } else {
        kernel_tensor->SetQuantType(kQT_Asym);
        kernel_tensor->SetZeroPoint(
            docWgt[(kernel_quant_name).c_str()]["zero_point"][0].GetFloat());
      }
    }
  }

  return outputs;
}

TensorVec VisVpuCodegen::Mean(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";  //
  auto input_name = inputs.at(0);

  TensorPtr input_tensor = graph->FindTensor(input_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, REDUCE_MEAN_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetOutputTensor(0, output_tensor);

  // parse attr
  const auto* attrs = call->attrs.as<ReduceAttrs>();  //

  CHECK(attrs) << "not ReduceAttrs node";  //

  auto axis = GetConcrete_int<int>(attrs->axis);
  bool keepdims = attrs->keepdims;

  // config param
  ReduceMeanOpPtr op_ptr = std::dynamic_pointer_cast<ReduceMeanOp>(node->GetOp());  //

  ReduceMeanParam& param = op_ptr->GetParam();  //

  param.axes[0] = axis[0];
  param.axes[1] = axis[1];
  param.keepdims = keepdims;

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  return outputs;
}

TensorVec VisVpuCodegen::Relu(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";  //
  auto input_name = inputs.at(0);

  TensorPtr input_tensor = graph->FindTensor(input_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, RELU_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetOutputTensor(0, output_tensor);

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  if (doTiling)
  {
    std::string tiling_name;
    tiling_name = op_name;
    int output_tile_num = docTiling[(tiling_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(tiling_name).c_str()]["output_dim_num"].GetInt();

    int output_tile_dim = docTiling[(tiling_name).c_str()]["output_tile_dims"][0].GetInt();
    std::vector<TileTensorShape> tile_shape_list;
    printf("***********relu************\n");
    printf("tilenum: %d ", output_tile_num);
    for (int k = 0; k < output_tile_num; k ++)
    {
      std::vector<int> tileShape;
      std::vector<int> tilePos;
      for (int i = 0; i < output_dim_num; ++i)
      {
        if (i == output_tile_dim)
        {
          tilePos.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
          tileShape.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - tilePos[i]);
        }
        else
        {
          assert(oshape[i] == docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt());
          tilePos.push_back(0);
          tileShape.push_back(oshape[i]);
        }
      }
      TileTensorShape tile_tensor_shape;
      tile_tensor_shape.shape = ShapeType(GetShapeType(tileShape));
      tile_tensor_shape.pos = ShapeType(GetShapeType(tilePos));
      tile_shape_list.push_back(tile_tensor_shape);
    }
    if (useTiling)
      RegisterTileSchema(op_name, tile_shape_list);

    int group_index = docTiling[(tiling_name).c_str()]["group_index"].GetInt();
    int call_index = docTiling[(tiling_name).c_str()]["call_index"].GetInt();
    auto find = tile_sort_map.find(group_index);
    if (find != tile_sort_map.end()) 
    {
      size_t insert = 0;
      for (; insert < tile_sort_map[group_index]->call_index_list.size(); insert++)
      {
        if (tile_sort_map[group_index]->call_index_list[insert] > call_index)
        {
          break;
        }
      }

      tile_sort_map[group_index]->call_index_list.insert(tile_sort_map[group_index]->call_index_list.begin() + insert, call_index);
      tile_sort_map[group_index]->call_name_list.insert(tile_sort_map[group_index]->call_name_list.begin() + insert, op_name);
    }
    else
    {
      GroupInfoPtr group_info = GroupInfoPtr(new GroupInfo);
      group_info->index = group_index;
      group_info->tile_num = output_tile_num;
      group_info->call_index_list.push_back(call_index);
      group_info->call_name_list.push_back(op_name);
      tile_sort_map[group_index] = group_info;
    }
  }

  if (doAllocate)
  {
    printf("***********doAllocate************\n");
    int output_tile_num = docTiling[(op_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(op_name).c_str()]["output_dim_num"].GetInt();
    // int output_tile_dim = docTiling[(op_name).c_str()]["output_tile_dims"][0].GetInt();
    std::string op_input_name = input_tensor->GetName();
    std::string op_output_name = output_tensor->GetName();

    for (int k = 0; k < output_tile_num; k++)
    {
      std::string tile_name = op_name + ".tile" + std::to_string(k);
      mem_allocator_builder_->CreateTile(tile_name, 0);

      // create input tensor
      printf("***********create input tensor************\n");
      std::string tile_input_name = op_input_name;
      std::vector<int> inputTileShape;
      std::vector<int> inputTilePos;
      for (int i = 0; i < output_dim_num; i++)
      {
        int index = docTiling[(op_name).c_str()]["input_tile_pos"].Size() == 1 ? 0 : k;
        inputTilePos.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][0].GetInt());
        inputTileShape.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][1].GetInt() - inputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_input_name += "@";
        else
          tile_input_name += ",";
        tile_input_name = tile_input_name + std::to_string(inputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_input_name += ":";
        else
          tile_input_name += ",";
        tile_input_name = tile_input_name + std::to_string(inputTileShape[i]);
      }
      int input_bit_width = docAct[(op_input_name).c_str()]["bit_width"].GetInt();
      int input_block_type = BLOCK_TYPE_INPUT;
      mem_allocator_builder_->CreateBlock(tile_name, op_input_name, 
          tile_input_name, inputTileShape, inputTilePos,
          input_bit_width/8, input_block_type);

      // create output tensor
      printf("***********create output tensor************\n");
      std::string tile_output_name = op_output_name;
      std::vector<int> outputTileShape;
      std::vector<int> outputTilePos;
      for (int i = 0; i < output_dim_num; i++)
      {
        outputTilePos.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
        outputTileShape.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - outputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_output_name += "@";
        else
          tile_output_name += ",";
        tile_output_name = tile_output_name + std::to_string(outputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_output_name += ":";
        else
          tile_output_name += ",";
        tile_output_name = tile_output_name + std::to_string(outputTileShape[i]);
      }
      int output_bit_width = docAct[(op_output_name).c_str()]["bit_width"].GetInt();
      int output_block_type = BLOCK_TYPE_OUTPUT;
      if (layer_idx + 1 == docTiling["network"]["call_num"].GetInt())
      {
        output_block_type = BLOCK_TYPE_NET_OUTPUT;
      }
      mem_allocator_builder_->CreateBlock(tile_name, op_output_name, 
          tile_output_name, outputTileShape, outputTilePos,
          output_bit_width/8, output_block_type);
      mem_allocator_builder_->CreateReuse(tile_name, tile_input_name, tile_output_name, false);

    }
  }

  return outputs;
}

TensorVec VisVpuCodegen::Reshape(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";  //
  auto input_name = inputs.at(0);

  TensorPtr input_tensor = graph->FindTensor(input_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, RESHAPE_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetOutputTensor(0, output_tensor);

  // parse attr
  const auto* attrs = call->attrs.as<ReshapeAttrs>();  //

  CHECK(attrs) << "not ReshapeAttrs node";  //

  auto newshape = GetConcrete_int<int>(attrs->newshape);

  // cal the real num when dim[i]==-1
  auto input_shape = input_tensor->GetShape();
  int shapes_all = 1;
  for (int i = 0; i < input_shape.dim_num; i++) shapes_all = shapes_all * input_shape.dim[i];

  int shapes_all_out = 1;
  int dim_num_ = -1;
  for (size_t i = 0; i < newshape.size(); i++) {
    if (newshape[i] == -1)
      dim_num_ = i;
    else
      shapes_all_out = shapes_all_out * newshape[i];
  }
  if (dim_num_ >= 0) {
    newshape[dim_num_] = shapes_all / shapes_all_out;
  }

  // config param
  ReshapeOpPtr op_ptr = std::dynamic_pointer_cast<ReshapeOp>(node->GetOp());  //
  if (newshape.size() == 4) {
    op_ptr->SetTargetShape(ShapeType(newshape[0], newshape[1], newshape[2], newshape[3]));
  }
  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  if (oshape.size() == 6 && oshape[0] == 1) {
    output_tensor->SetShape(ShapeType(oshape[1], oshape[2], oshape[3], oshape[4], oshape[5]));
  } else {
    output_tensor->SetShape(GetShapeType(oshape));
  }

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  if (doTiling)
  {
    std::string tiling_name;
    tiling_name = op_name;
    bool need_tiling = true;
    if (!docTiling[(tiling_name).c_str()]["need_tiling"].GetBool())
    {
      useAllocate = false;
      need_tiling = false;
    }
    int output_tile_num = docTiling[(tiling_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(tiling_name).c_str()]["output_dim_num"].GetInt();
    assert(output_dim_num == 4);
    int output_tile_dim = docTiling[(tiling_name).c_str()]["output_tile_dims"][0].GetInt();
    std::vector<TileTensorShape> tile_shape_list;
    for (int k = 0; k < output_tile_num; k ++)
    {
      std::vector<int> tileShape;
      std::vector<int> tilePos;
      for (int i = 0; i < output_dim_num; ++i)
      {
        if (i == output_tile_dim)
        {
          tilePos.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
          tileShape.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - tilePos[i]);
        }
        else
        {
          assert(oshape[i] == docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt());
          tilePos.push_back(0);
          tileShape.push_back(oshape[i]);
        }
      }
      TileTensorShape tile_tensor_shape;
      tile_tensor_shape.shape = ShapeType(GetShapeType(tileShape));
      tile_tensor_shape.pos = ShapeType(GetShapeType(tilePos));
      tile_shape_list.push_back(tile_tensor_shape);
    }
    if (need_tiling && useTiling)
    {
      RegisterTileSchema(op_name, tile_shape_list);
    }

    int group_index = docTiling[(tiling_name).c_str()]["group_index"].GetInt();
    int call_index = docTiling[(tiling_name).c_str()]["call_index"].GetInt();
    auto find = tile_sort_map.find(group_index);
    if (find != tile_sort_map.end()) 
    {
      size_t insert = 0;
      for (; insert < tile_sort_map[group_index]->call_index_list.size(); insert++)
      {
        if (tile_sort_map[group_index]->call_index_list[insert] > call_index)
        {
          break;
        }
      }

      tile_sort_map[group_index]->call_index_list.insert(tile_sort_map[group_index]->call_index_list.begin() + insert, call_index);
      tile_sort_map[group_index]->call_name_list.insert(tile_sort_map[group_index]->call_name_list.begin() + insert, op_name);
      tile_sort_map[group_index]->need_tiling = need_tiling;
    }
    else
    {
      GroupInfoPtr group_info = GroupInfoPtr(new GroupInfo);
      group_info->index = group_index;
      group_info->tile_num = output_tile_num;
      group_info->call_index_list.push_back(call_index);
      group_info->call_name_list.push_back(op_name);
      group_info->need_tiling = need_tiling;
      tile_sort_map[group_index] = group_info;
    }
  }

  if (doAllocate)
  {
    printf("***********doAllocate************\n");
    int output_tile_num = docTiling[(op_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(op_name).c_str()]["output_dim_num"].GetInt();
    int input_dim_num = docTiling[(op_name).c_str()]["input_dim_num"].GetInt();
    // int output_tile_dim = docTiling[(op_name).c_str()]["output_tile_dims"][0].GetInt();
    std::string op_input_name = input_tensor->GetName();
    std::string op_output_name = output_tensor->GetName();

    for (int k = 0; k < output_tile_num; k++)
    {
      std::string tile_name = op_name + ".tile" + std::to_string(k);
      mem_allocator_builder_->CreateTile(tile_name, 0);

      // create input tensor
      printf("***********create input tensor************\n");
      std::string tile_input_name = op_input_name;
      std::vector<int> inputTileShape;
      std::vector<int> inputTilePos;
      for (int i = 0; i < input_dim_num; i++)
      {
        int index = docTiling[(op_name).c_str()]["input_tile_pos"].Size() == 1 ? 0 : k;
        inputTilePos.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][0].GetInt());
        inputTileShape.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][1].GetInt() - inputTilePos[i]);
      }
      for (int i = 0; i < input_dim_num; i++)
      {
        if (i==0)
          tile_input_name += "@";
        else
          tile_input_name += ",";
        tile_input_name = tile_input_name + std::to_string(inputTilePos[i]);
      }
      for (int i = 0; i < input_dim_num; i++)
      {
        if (i==0)
          tile_input_name += ":";
        else
          tile_input_name += ",";
        tile_input_name = tile_input_name + std::to_string(inputTileShape[i]);
      }
      int input_bit_width = docAct[(op_input_name).c_str()]["bit_width"].GetInt();
      int input_block_type = BLOCK_TYPE_INPUT;
      mem_allocator_builder_->CreateBlock(tile_name, op_input_name, 
          tile_input_name, inputTileShape, inputTilePos,
          input_bit_width/8, input_block_type);

      // create output tensor
      printf("***********create output tensor************\n");
      std::string tile_output_name = op_output_name;
      std::vector<int> outputTileShape;
      std::vector<int> outputTilePos;
      for (int i = 0; i < output_dim_num; i++)
      {
        outputTilePos.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
        outputTileShape.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - outputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_output_name += "@";
        else
          tile_output_name += ",";
        tile_output_name = tile_output_name + std::to_string(outputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_output_name += ":";
        else
          tile_output_name += ",";
        tile_output_name = tile_output_name + std::to_string(outputTileShape[i]);
      }
      int output_bit_width = docAct[(op_output_name).c_str()]["bit_width"].GetInt();
      int output_block_type = BLOCK_TYPE_OUTPUT;
      if (layer_idx + 1 == docTiling["network"]["call_num"].GetInt())
      {
        output_block_type = BLOCK_TYPE_NET_OUTPUT;
      }
      mem_allocator_builder_->CreateBlock(tile_name, op_output_name, 
          tile_output_name, outputTileShape, outputTilePos,
          output_bit_width/8, output_block_type);
      
      mem_allocator_builder_->CreateReuse(tile_name, tile_input_name, tile_output_name, true);

    }
  }

  return outputs;
}

TensorVec VisVpuCodegen::Resize(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";  //
  auto input_name = inputs.at(0);

  TensorPtr input_tensor = graph->FindTensor(input_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, RESIZE_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetOutputTensor(0, output_tensor);

  // parse attr
  const auto* attrs = call->attrs.as<Resize2DAttrs>();  //
  std::cout << "Resize attrs==" << attrs << std::endl;

  CHECK(attrs) << "not Resize2DAttrs node";  //

  std::string method = attrs->method;

  // config param
  ResizeOpPtr op_ptr = std::dynamic_pointer_cast<ResizeOp>(node->GetOp());  //

  // if (method == "nearest_neighbor") {
  //     op_ptr->SetMode(0);
  // } else if (method == "linear") {
  //     op_ptr->SetMode(1);
  // } else if (method == "cubic") {
  //     op_ptr->SetMode(2);
  // } else {
  //     LOG(FATAL) << "Unknown method";
  // }

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  if (doTiling)
  {
    std::string tiling_name;
    tiling_name = op_name;
    int output_tile_num = docTiling[(tiling_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(tiling_name).c_str()]["output_dim_num"].GetInt();

    int output_tile_dim = docTiling[(tiling_name).c_str()]["output_tile_dims"][0].GetInt();
    std::vector<TileTensorShape> tile_shape_list;
    for (int k = 0; k < output_tile_num; k ++)
    {
      std::vector<int> tileShape;
      std::vector<int> tilePos;
      for (int i = 0; i < output_dim_num; ++i)
      {
        if (i == output_tile_dim)
        {
          tilePos.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
          tileShape.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - tilePos[i]);
        }
        else
        {
          assert(oshape[i] == docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt());
          tilePos.push_back(0);
          tileShape.push_back(oshape[i]);
        }
      }
      TileTensorShape tile_tensor_shape;
      tile_tensor_shape.shape = ShapeType(GetShapeType(tileShape));
      tile_tensor_shape.pos = ShapeType(GetShapeType(tilePos));
      tile_shape_list.push_back(tile_tensor_shape);
    }
    if (useTiling)
      RegisterTileSchema(op_name, tile_shape_list);

    int group_index = docTiling[(tiling_name).c_str()]["group_index"].GetInt();
    int call_index = docTiling[(tiling_name).c_str()]["call_index"].GetInt();
    auto find = tile_sort_map.find(group_index);
    if (find != tile_sort_map.end()) 
    {
      size_t insert = 0;
      for (; insert < tile_sort_map[group_index]->call_index_list.size(); insert++)
      {
        if (tile_sort_map[group_index]->call_index_list[insert] > call_index)
        {
          break;
        }
      }

      tile_sort_map[group_index]->call_index_list.insert(tile_sort_map[group_index]->call_index_list.begin() + insert, call_index);
      tile_sort_map[group_index]->call_name_list.insert(tile_sort_map[group_index]->call_name_list.begin() + insert, op_name);
    }
    else
    {
      GroupInfoPtr group_info = GroupInfoPtr(new GroupInfo);
      group_info->index = group_index;
      group_info->tile_num = output_tile_num;
      group_info->call_index_list.push_back(call_index);
      group_info->call_name_list.push_back(op_name);
      tile_sort_map[group_index] = group_info;
    }
  }

  if (doAllocate)
  {
    printf("***********doAllocate************\n");
    int output_tile_num = docTiling[(op_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(op_name).c_str()]["output_dim_num"].GetInt();
    // int output_tile_dim = docTiling[(op_name).c_str()]["output_tile_dims"][0].GetInt();
    std::string op_input_name = input_tensor->GetName();
    std::string op_output_name = output_tensor->GetName();

    for (int k = 0; k < output_tile_num; k++)
    {
      std::string tile_name = op_name + ".tile" + std::to_string(k);
      mem_allocator_builder_->CreateTile(tile_name, 0);

      // create input tensor
      printf("***********create input tensor************\n");
      std::string tile_input_name = op_input_name;
      std::vector<int> inputTileShape;
      std::vector<int> inputTilePos;
      for (int i = 0; i < output_dim_num; i++)
      {
        int index = docTiling[(op_name).c_str()]["input_tile_pos"].Size() == 1 ? 0 : k;
        inputTilePos.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][0].GetInt());
        inputTileShape.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][1].GetInt() - inputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_input_name += "@";
        else
          tile_input_name += ",";
        tile_input_name = tile_input_name + std::to_string(inputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_input_name += ":";
        else
          tile_input_name += ",";
        tile_input_name = tile_input_name + std::to_string(inputTileShape[i]);
      }
      int input_bit_width = docAct[(op_input_name).c_str()]["bit_width"].GetInt();
      int input_block_type = BLOCK_TYPE_INPUT;
      mem_allocator_builder_->CreateBlock(tile_name, op_input_name, 
          tile_input_name, inputTileShape, inputTilePos,
          input_bit_width/8, input_block_type);

      // create output tensor
      printf("***********create output tensor************\n");
      std::string tile_output_name = op_output_name;
      std::vector<int> outputTileShape;
      std::vector<int> outputTilePos;
      for (int i = 0; i < output_dim_num; i++)
      {
        outputTilePos.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
        outputTileShape.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - outputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_output_name += "@";
        else
          tile_output_name += ",";
        tile_output_name = tile_output_name + std::to_string(outputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_output_name += ":";
        else
          tile_output_name += ",";
        tile_output_name = tile_output_name + std::to_string(outputTileShape[i]);
      }
      int output_bit_width = docAct[(op_output_name).c_str()]["bit_width"].GetInt();
      int output_block_type = BLOCK_TYPE_OUTPUT;
      if (layer_idx + 1 == docTiling["network"]["call_num"].GetInt())
      {
        output_block_type = BLOCK_TYPE_NET_OUTPUT;
      }
      mem_allocator_builder_->CreateBlock(tile_name, op_output_name, 
          tile_output_name, outputTileShape, outputTilePos,
          output_bit_width/8, output_block_type);

    }
  }

  return outputs;
}

TensorVec VisVpuCodegen::Pad(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  // CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";  //
  auto input_name = inputs.at(0);

  TensorPtr input_tensor = graph->FindTensor(input_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, PAD_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetOutputTensor(0, output_tensor);

  // parse attr
  const auto* attrs = call->attrs.as<PadAttrs>();  //
  std::cout << "Pad attrs==" << attrs << std::endl;

  CHECK(attrs) << "not PadAttrs node";  //

  // auto padding = GetConcrete<int>(attrs->pad_width);

  std::vector<int> pad_width_vec;
  for (const auto& arr : attrs->pad_width) {
    for (const auto& val : arr) {
      pad_width_vec.push_back(static_cast<int>(val.as<IntImmNode>()->value));
    }
  }

  // config param
  PadOpPtr op_ptr = std::dynamic_pointer_cast<PadOp>(node->GetOp());  //

  PadParam& param = op_ptr->GetParam();
  param.mode = 0;

  for (int i = 0; i < 8; i++) {
    param.pad[i] = pad_width_vec[i];
  }

  // some onnx file with 2 inputs for nn.pad
  if (inputs.size() == 2) {
    // pad [s1, s2, s3, s4, e1, e2, e3, e4] ---- pad_width[s1, e1, s2, e2, s3,e3, s4, e4]
    int i = 0;
    for (i = 0; i < 4; i++) {
      param.pad[i] = pad_width_vec[i * 2];
    }

    for (i = 4; i < 8; i++) {
      param.pad[i] = pad_width_vec[i * 2 - 7];
    }
  }

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  return outputs;
}

TensorVec VisVpuCodegen::Upsample(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);
  ////	Resize(const OpBuildArgs& op_build_args);
  // create node
  CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";  //
  auto input_name = inputs.at(0);

  TensorPtr input_tensor = graph->FindTensor(input_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, UPSAMPLE_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetOutputTensor(0, output_tensor);

  // parse attr
  const auto* attrs = call->attrs.as<Resize2DAttrs>();  //
  std::cout << "upsample attrs==" << attrs << std::endl;

  //		CHECK(attrs) << "not Resize2DAttrs node";//

  //		std::string method = attrs->mode;

  // config param
  ReshapeOpPtr op_ptr = std::dynamic_pointer_cast<ReshapeOp>(node->GetOp());  //

  //		 if (method == "nearest_neighbor") {
  //			   op_ptr->SetMode(0);
  //		 } else if (method == "linear") {
  //			   op_ptr->SetMode(1);
  //		 } else if (method == "cubic") {
  //			   op_ptr->SetMode(2);
  //		 } else {
  //			   LOG(FATAL) << "Unknown method";
  //		 }

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  return outputs;
}

TensorVec VisVpuCodegen::Sigmoid(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";  //
  auto input_name = inputs.at(0);

  TensorPtr input_tensor = graph->FindTensor(input_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, SIGMOD_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetOutputTensor(0, output_tensor);

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  if (doTiling)
  {
    std::string tiling_name;
    tiling_name = op_name;
    int output_tile_num = docTiling[(tiling_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(tiling_name).c_str()]["output_dim_num"].GetInt();

    int output_tile_dim = docTiling[(tiling_name).c_str()]["output_tile_dims"][0].GetInt();
    std::vector<TileTensorShape> tile_shape_list;
    for (int k = 0; k < output_tile_num; k ++)
    {
      std::vector<int> tileShape;
      std::vector<int> tilePos;
      for (int i = 0; i < output_dim_num; ++i)
      {
        if (i == output_tile_dim)
        {
          tilePos.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
          tileShape.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - tilePos[i]);
        }
        else
        {
          assert(oshape[i] == docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt());
          tilePos.push_back(0);
          tileShape.push_back(oshape[i]);
        }
      }
      TileTensorShape tile_tensor_shape;
      tile_tensor_shape.shape = ShapeType(GetShapeType(tileShape));
      tile_tensor_shape.pos = ShapeType(GetShapeType(tilePos));
      tile_shape_list.push_back(tile_tensor_shape);
    }
    if (useTiling)
      RegisterTileSchema(op_name, tile_shape_list);

    int group_index = docTiling[(tiling_name).c_str()]["group_index"].GetInt();
    int call_index = docTiling[(tiling_name).c_str()]["call_index"].GetInt();
    auto find = tile_sort_map.find(group_index);
    if (find != tile_sort_map.end()) 
    {
      size_t insert = 0;
      for (; insert < tile_sort_map[group_index]->call_index_list.size(); insert++)
      {
        if (tile_sort_map[group_index]->call_index_list[insert] > call_index)
        {
          break;
        }
      }

      tile_sort_map[group_index]->call_index_list.insert(tile_sort_map[group_index]->call_index_list.begin() + insert, call_index);
      tile_sort_map[group_index]->call_name_list.insert(tile_sort_map[group_index]->call_name_list.begin() + insert, op_name);
    }
    else
    {
      GroupInfoPtr group_info = GroupInfoPtr(new GroupInfo);
      group_info->index = group_index;
      group_info->tile_num = output_tile_num;
      group_info->call_index_list.push_back(call_index);
      group_info->call_name_list.push_back(op_name);
      tile_sort_map[group_index] = group_info;
    }
  }

  return outputs;
}

TensorVec VisVpuCodegen::Slice(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";  //
  auto input_name = inputs.at(0);

  TensorPtr input_tensor = graph->FindTensor(input_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, SLICE_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetOutputTensor(0, output_tensor);
  // parse attr
  const auto* attrs = call->attrs.as<StridedSliceAttrs>();  //

  CHECK(attrs && attrs->begin && attrs->end && attrs->strides && attrs->axes)
      << "not StridedSliceAttrs node";  //

  auto axes = GetConcrete_int<int>(attrs->axes.value());
  auto begin = GetConcrete_int<int>(attrs->begin.value());
  auto end = GetConcrete_int<int>(attrs->end.value());
  auto strides = GetConcrete_int<int>(attrs->strides.value());

  // config param
  SliceOpPtr op_ptr = std::dynamic_pointer_cast<SliceOp>(node->GetOp());  //

  SliceParam& param = op_ptr->GetParam();  //

  auto in_shape = input_tensor->GetShape();

  for (size_t i = 0; i < axes.size(); i++) {
    if (end[i] == -1) {
      param.ends.push_back(in_shape.dim[axes[i]]);
    } else if (end[i] < -1) {
      param.ends.push_back(in_shape.dim[axes[i]] + end[i]);
    } else {
      param.ends.push_back(end[i]);
    }
    if (begin[i] < 0) {
      param.starts.push_back(in_shape.dim[axes[i]] + begin[i]);
    } else {
      param.starts.push_back(begin[i]);
    }
    param.axes.push_back(axes[i]);
    param.steps.push_back(strides[i]);
  }

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  if (doTiling)
  {
    std::string tiling_name;
    tiling_name = op_name;
    bool need_tiling = true;
    if (!docTiling[(tiling_name).c_str()]["need_tiling"].GetBool())
    {
      useAllocate = false;
      need_tiling = false;
    }
    int output_tile_num = docTiling[(tiling_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(tiling_name).c_str()]["output_dim_num"].GetInt();
    assert(output_dim_num == 4);
    int output_tile_dim = docTiling[(tiling_name).c_str()]["output_tile_dims"][0].GetInt();
    std::vector<TileTensorShape> tile_shape_list;
    for (int k = 0; k < output_tile_num; k ++)
    {
      std::vector<int> tileShape;
      std::vector<int> tilePos;
      for (int i = 0; i < output_dim_num; ++i)
      {
        if (i == output_tile_dim)
        {
          tilePos.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
          tileShape.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - tilePos[i]);
        }
        else
        {
          assert(oshape[i] == docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt());
          tilePos.push_back(0);
          tileShape.push_back(oshape[i]);
        }
      }
      TileTensorShape tile_tensor_shape;
      tile_tensor_shape.shape = ShapeType(GetShapeType(tileShape));
      tile_tensor_shape.pos = ShapeType(GetShapeType(tilePos));
      tile_shape_list.push_back(tile_tensor_shape);
    }
    if (need_tiling && useTiling)
    {
      RegisterTileSchema(op_name, tile_shape_list);
    }

    int group_index = docTiling[(tiling_name).c_str()]["group_index"].GetInt();
    int call_index = docTiling[(tiling_name).c_str()]["call_index"].GetInt();
    auto find = tile_sort_map.find(group_index);
    if (find != tile_sort_map.end()) 
    {
      size_t insert = 0;
      for (; insert < tile_sort_map[group_index]->call_index_list.size(); insert++)
      {
        if (tile_sort_map[group_index]->call_index_list[insert] > call_index)
        {
          break;
        }
      }

      tile_sort_map[group_index]->call_index_list.insert(tile_sort_map[group_index]->call_index_list.begin() + insert, call_index);
      tile_sort_map[group_index]->call_name_list.insert(tile_sort_map[group_index]->call_name_list.begin() + insert, op_name);
      tile_sort_map[group_index]->need_tiling = need_tiling;
    }
    else
    {
      GroupInfoPtr group_info = GroupInfoPtr(new GroupInfo);
      group_info->index = group_index;
      group_info->tile_num = output_tile_num;
      group_info->call_index_list.push_back(call_index);
      group_info->call_name_list.push_back(op_name);
      group_info->need_tiling = need_tiling;
      tile_sort_map[group_index] = group_info;
    }
  }

  if (doAllocate)
  {
    printf("***********doAllocate************\n");
    int output_tile_num = docTiling[(op_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(op_name).c_str()]["output_dim_num"].GetInt();
    int input_dim_num = docTiling[(op_name).c_str()]["input_dim_num"].GetInt();
    // int output_tile_dim = docTiling[(op_name).c_str()]["output_tile_dims"][0].GetInt();
    std::string op_input_name = input_tensor->GetName();
    std::string op_output_name = output_tensor->GetName();

    for (int k = 0; k < output_tile_num; k++)
    {
      std::string tile_name = op_name + ".tile" + std::to_string(k);
      mem_allocator_builder_->CreateTile(tile_name, 0);

      // create input tensor
      printf("***********create input tensor************\n");
      std::string tile_input_name = op_input_name;
      std::vector<int> inputTileShape;
      std::vector<int> inputTilePos;
      for (int i = 0; i < input_dim_num; i++)
      {
        int index = docTiling[(op_name).c_str()]["input_tile_pos"].Size() == 1 ? 0 : k;
        inputTilePos.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][0].GetInt());
        inputTileShape.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][1].GetInt() - inputTilePos[i]);
      }
      for (int i = 0; i < input_dim_num; i++)
      {
        if (i==0)
          tile_input_name += "@";
        else
          tile_input_name += ",";
        tile_input_name = tile_input_name + std::to_string(inputTilePos[i]);
      }
      for (int i = 0; i < input_dim_num; i++)
      {
        if (i==0)
          tile_input_name += ":";
        else
          tile_input_name += ",";
        tile_input_name = tile_input_name + std::to_string(inputTileShape[i]);
      }
      int input_bit_width = docAct[(op_input_name).c_str()]["bit_width"].GetInt();
      int input_block_type = BLOCK_TYPE_INPUT;
      mem_allocator_builder_->CreateBlock(tile_name, op_input_name, 
          tile_input_name, inputTileShape, inputTilePos,
          input_bit_width/8, input_block_type);

      // create output tensor
      printf("***********create output tensor************\n");
      std::string tile_output_name = op_output_name;
      std::vector<int> outputTileShape;
      std::vector<int> outputTilePos;
      for (int i = 0; i < output_dim_num; i++)
      {
        outputTilePos.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
        outputTileShape.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - outputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_output_name += "@";
        else
          tile_output_name += ",";
        tile_output_name = tile_output_name + std::to_string(outputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_output_name += ":";
        else
          tile_output_name += ",";
        tile_output_name = tile_output_name + std::to_string(outputTileShape[i]);
      }
      int output_bit_width = docAct[(op_output_name).c_str()]["bit_width"].GetInt();
      int output_block_type = BLOCK_TYPE_OUTPUT;
      if (layer_idx + 1 == docTiling["network"]["call_num"].GetInt())
      {
        output_block_type = BLOCK_TYPE_NET_OUTPUT;
      }
      mem_allocator_builder_->CreateBlock(tile_name, op_output_name, 
          tile_output_name, outputTileShape, outputTilePos,
          output_bit_width/8, output_block_type);
    }
  }

  decltype(inputs) outputs{op_name + ":out"};

  return outputs;
}

TensorVec VisVpuCodegen::Split(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";  //
  auto input_name = inputs.at(0);

  TensorPtr input_tensor = graph->FindTensor(input_name);

  NodePtr node = graph->CreateNode(op_name, SPLIT_OP);  //

  node->SetInputTensor(0, input_tensor);

  int numoutputs = 1;

  std::vector<std::string> outputs;

  // parse attr
  const auto* attrs = call->attrs.as<SplitAttrs>();  //

  CHECK(attrs) << "not SplitAttrs node";  //

  int axis = attrs->axis;

  if (const IntImmNode* sections = attrs->indices_or_sections.as<IntImmNode>()) {
    numoutputs = sections->value;
  } else {
    auto indices = Downcast<tvm::Array<Integer>>(attrs->indices_or_sections);
    numoutputs = indices.size() + 1;
  }

  // config param
  SplitOpPtr op_ptr = std::dynamic_pointer_cast<SplitOp>(node->GetOp());  //

  op_ptr->SetAxis(axis);
  op_ptr->SetOutputNumber(numoutputs);

  for (int32_t i = 0; i < numoutputs; i++) {
    vis::megrez::DataType out_type = Get_split_type(op_name, i);  // todo
    TensorPtr output_tensor =
        graph->CreateTensor(op_name + ":out" + "_" + std::to_string(i), kTT_Var, out_type);
    node->SetOutputTensor(i, output_tensor);

    // quant output
    output_tensor->SetScale(
        docAct[(op_name + ":out" + "_" + std::to_string(i)).c_str()]["scale"][0].GetFloat());

    output_tensor->SetMax(
        docAct[(op_name + ":out" + "_" + std::to_string(i)).c_str()]["max_value"][0].GetFloat());
    output_tensor->SetMin(
        docAct[(op_name + ":out" + "_" + std::to_string(i)).c_str()]["min_value"][0].GetFloat());
    if (std::strcmp(
            docAct[(op_name + ":out" + "_" + std::to_string(i)).c_str()]["quantizer"].GetString(),
            "Symmetric") == 0) {
      output_tensor->SetQuantType(kQT_Sym);
      output_tensor->SetZeroPoint(
          docAct[(op_name + ":out" + "_" + std::to_string(i)).c_str()]["zero_point"].GetFloat());
    } else {
      output_tensor->SetQuantType(kQT_Asym);
      output_tensor->SetZeroPoint(
          docAct[(op_name + ":out" + "_" + std::to_string(i)).c_str()]["zero_point"][0].GetFloat());
    }

    outputs.push_back(op_name + ":out" + "_" + std::to_string(i));  /// todo
  }

  // outputs.push_back(op_name + ":out");

  return outputs;
}

TensorVec VisVpuCodegen::Transpose(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";  //
  auto input_name = inputs.at(0);

  TensorPtr input_tensor = graph->FindTensor(input_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, TRANSPOSE_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetOutputTensor(0, output_tensor);

  // parse attr
  const auto* attrs = call->attrs.as<TransposeAttrs>();  //

  CHECK(attrs) << "not TransposeAttrs node";  //

  auto axes = GetConcrete_int<int>(attrs->axes);

  // config param
  TransposeOpPtr op_ptr = std::dynamic_pointer_cast<TransposeOp>(node->GetOp());  //

  TransposeParam& param = op_ptr->GetParam();  //

  if (axes.size() == 6) {
    for (size_t idx = 1; idx < axes.size(); ++idx) {
      param.perm.push_back(axes[idx] - 1);
    }
  } else {
    for (size_t idx = 0; idx < axes.size(); ++idx) {
      param.perm.push_back(axes[idx]);
    }
  }

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  // output_tensor->SetShape(GetShapeType(oshape));
  if (oshape.size() == 6 && oshape[0] == 1) {
    output_tensor->SetShape(ShapeType(oshape[1], oshape[2], oshape[3], oshape[4], oshape[5]));
  } else {
    output_tensor->SetShape(GetShapeType(oshape));
  }

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  if (doTiling)
  {
    std::string tiling_name;
    tiling_name = op_name;
    bool need_tiling = true;
    if (!docTiling[(tiling_name).c_str()]["need_tiling"].GetBool())
    {
      useAllocate = false;
      need_tiling = false;
    }
    int output_tile_num = docTiling[(tiling_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(tiling_name).c_str()]["output_dim_num"].GetInt();
    assert(output_dim_num == 4);
    int output_tile_dim = docTiling[(tiling_name).c_str()]["output_tile_dims"][0].GetInt();
    std::vector<TileTensorShape> tile_shape_list;
    for (int k = 0; k < output_tile_num; k ++)
    {
      std::vector<int> tileShape;
      std::vector<int> tilePos;
      for (int i = 0; i < output_dim_num; ++i)
      {
        if (i == output_tile_dim)
        {
          tilePos.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
          tileShape.push_back(docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - tilePos[i]);
        }
        else
        {
          assert(oshape[i] == docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt());
          tilePos.push_back(0);
          tileShape.push_back(oshape[i]);
        }
      }
      TileTensorShape tile_tensor_shape;
      tile_tensor_shape.shape = ShapeType(GetShapeType(tileShape));
      tile_tensor_shape.pos = ShapeType(GetShapeType(tilePos));
      tile_shape_list.push_back(tile_tensor_shape);
    }
    if (need_tiling && useTiling)
    {
      RegisterTileSchema(op_name, tile_shape_list);
    }

    int group_index = docTiling[(tiling_name).c_str()]["group_index"].GetInt();
    int call_index = docTiling[(tiling_name).c_str()]["call_index"].GetInt();
    auto find = tile_sort_map.find(group_index);
    if (find != tile_sort_map.end()) 
    {
      size_t insert = 0;
      for (; insert < tile_sort_map[group_index]->call_index_list.size(); insert++)
      {
        if (tile_sort_map[group_index]->call_index_list[insert] > call_index)
        {
          break;
        }
      }

      tile_sort_map[group_index]->call_index_list.insert(tile_sort_map[group_index]->call_index_list.begin() + insert, call_index);
      tile_sort_map[group_index]->call_name_list.insert(tile_sort_map[group_index]->call_name_list.begin() + insert, op_name);
      tile_sort_map[group_index]->need_tiling = need_tiling;
    }
    else
    {
      GroupInfoPtr group_info = GroupInfoPtr(new GroupInfo);
      group_info->index = group_index;
      group_info->tile_num = output_tile_num;
      group_info->call_index_list.push_back(call_index);
      group_info->call_name_list.push_back(op_name);
      group_info->need_tiling = need_tiling;
      tile_sort_map[group_index] = group_info;
    }
  }

  if (doAllocate)
  {
    printf("***********doAllocate************\n");
    int output_tile_num = docTiling[(op_name).c_str()]["output_tile_num"].GetInt();
    int output_dim_num = docTiling[(op_name).c_str()]["output_dim_num"].GetInt();
    int input_dim_num = docTiling[(op_name).c_str()]["input_dim_num"].GetInt();
    // int output_tile_dim = docTiling[(op_name).c_str()]["output_tile_dims"][0].GetInt();
    std::string op_input_name = input_tensor->GetName();
    std::string op_output_name = output_tensor->GetName();

    for (int k = 0; k < output_tile_num; k++)
    {
      std::string tile_name = op_name + ".tile" + std::to_string(k);
      mem_allocator_builder_->CreateTile(tile_name, 0);

      // create input tensor
      printf("***********create input tensor************\n");
      std::string tile_input_name = op_input_name;
      std::vector<int> inputTileShape;
      std::vector<int> inputTilePos;
      for (int i = 0; i < input_dim_num; i++)
      {
        int index = docTiling[(op_name).c_str()]["input_tile_pos"].Size() == 1 ? 0 : k;
        inputTilePos.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][0].GetInt());
        inputTileShape.push_back(docTiling[(op_name).c_str()]["input_tile_pos"][index][i][1].GetInt() - inputTilePos[i]);
      }
      for (int i = 0; i < input_dim_num; i++)
      {
        if (i==0)
          tile_input_name += "@";
        else
          tile_input_name += ",";
        tile_input_name = tile_input_name + std::to_string(inputTilePos[i]);
      }
      for (int i = 0; i < input_dim_num; i++)
      {
        if (i==0)
          tile_input_name += ":";
        else
          tile_input_name += ",";
        tile_input_name = tile_input_name + std::to_string(inputTileShape[i]);
      }
      int input_bit_width = docAct[(op_input_name).c_str()]["bit_width"].GetInt();
      int input_block_type = BLOCK_TYPE_INPUT;
      mem_allocator_builder_->CreateBlock(tile_name, op_input_name, 
          tile_input_name, inputTileShape, inputTilePos,
          input_bit_width/8, input_block_type);

      // create output tensor
      printf("***********create output tensor************\n");
      std::string tile_output_name = op_output_name;
      std::vector<int> outputTileShape;
      std::vector<int> outputTilePos;
      for (int i = 0; i < output_dim_num; i++)
      {
        outputTilePos.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][0].GetInt());
        outputTileShape.push_back(docTiling[(op_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - outputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_output_name += "@";
        else
          tile_output_name += ",";
        tile_output_name = tile_output_name + std::to_string(outputTilePos[i]);
      }
      for (int i = 0; i < output_dim_num; i++)
      {
        if (i==0)
          tile_output_name += ":";
        else
          tile_output_name += ",";
        tile_output_name = tile_output_name + std::to_string(outputTileShape[i]);
      }
      int output_bit_width = docAct[(op_output_name).c_str()]["bit_width"].GetInt();
      int output_block_type = BLOCK_TYPE_OUTPUT;
      if (layer_idx + 1 == docTiling["network"]["call_num"].GetInt())
      {
        output_block_type = BLOCK_TYPE_NET_OUTPUT;
      }
      mem_allocator_builder_->CreateBlock(tile_name, op_output_name, 
          tile_output_name, outputTileShape, outputTilePos,
          output_bit_width/8, output_block_type);
    }
  }

  return outputs;
}

TensorVec VisVpuCodegen::Dense(const OpBuildArgs& op_build_args) {
  UNPACK_OP_BUILD_ARGS(op_build_args);
  // create node
  CHECK_EQ(inputs.size(), 2) << "number of inputs need to corrected";
  auto input_name = inputs.at(0);
  auto kernel_name = inputs.at(1);

  TensorPtr input_tensor = graph->FindTensor(input_name);
  TensorPtr kernel_tensor = graph->FindTensor(kernel_name);

  auto kernel_shape = kernel_tensor->GetShape();

  // fc or matmul
  auto input_shape = input_tensor->GetShape();
  int nozero = 0;
  for (int i = 0; i < input_shape.dim_num; i++) {
    if (input_shape.dim[i] > 1) {
      nozero++;
    }
  }
  bool isfc = (nozero == 1);

  // create name
  std::string op_name_org = op_name;
  std::string op_name_new = (isfc ? "fc_" : "matmul_") + std::to_string(layer_idx);

  vis::megrez::DataType out_type = Get_quant_type(op_name_org);
  TensorPtr output_tensor = graph->CreateTensor(op_name_new + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name_new, isfc ? FC_OP : MATMUL_OP);  //

  if (!isfc) {
    CHECK_EQ(kernel_shape.dim_num, 2) << "support 2 dims const matmul only";

    // cvt weight from N*K in tvm to K*N
    kernel_tensor->SetShape(ShapeType(kernel_shape.dim[1], kernel_shape.dim[0]));

    // Define the original dimensions of the weight data
    const int original_dims[2] = {kernel_shape.dim[0], kernel_shape.dim[1]};

    int new_weight_mem_size = original_dims[0] * original_dims[1];

    // Create a new array to hold the transposed weight data
    if (out_type == kDT_Int16 || out_type == kDT_Uint16) {
      uint16_t* transposed_data = new uint16_t[new_weight_mem_size];
      uint16_t* weight_data = (uint16_t*)kernel_tensor->GetData();
      // Loop through each element in the original weight data
      for (int i = 0; i < original_dims[1]; i++) {
        for (int j = 0; j < original_dims[0]; j++) {
          // Calculate the index of the current element in the transposed array
          int original_index = j * original_dims[1] + i;
          // Calculate the index of the current element in the original array
          int transposed_index = i * original_dims[0] + j;
          // Copy the current element from the original array to the transposed array
          transposed_data[transposed_index] = weight_data[original_index];
        }
      }
      // int8_t* int_data = reinterpret_cast<int8_t*>(transposed_data);
      // Use the transposed weight data as needed
      kernel_tensor->SetData((uint8_t*)transposed_data, new_weight_mem_size * 2);

    } else {
      uint8_t* transposed_data = new uint8_t[new_weight_mem_size];
      uint8_t* weight_data = kernel_tensor->GetData();
      // Loop through each element in the original weight data
      for (int i = 0; i < original_dims[1]; i++) {
        for (int j = 0; j < original_dims[0]; j++) {
          // Calculate the index of the current element in the transposed array
          int original_index = j * original_dims[1] + i;
          // Calculate the index of the current element in the original array
          int transposed_index = i * original_dims[0] + j;
          // Copy the current element from the original array to the transposed array
          transposed_data[transposed_index] = weight_data[original_index];
        }
      }
      // int8_t* int_data = reinterpret_cast<int8_t*>(transposed_data);
      // Use the transposed weight data as needed
      kernel_tensor->SetData(transposed_data, new_weight_mem_size);
    }
  }
  node->SetInputTensor(0, input_tensor);
  node->SetInputTensor(1, kernel_tensor);
  node->SetOutputTensor(0, output_tensor);

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name_org + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name_org + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name_org + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name_org + ":out").c_str()]["quantizer"].GetString(), "Symmetric") ==
      0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name_org + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name_org + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name_new + ":out"};

  // quant weight
  auto op_node = call->op.as<OpNode>();
  auto operator_id = op_node->name;
  std::string kernel_quant_name;
  kernel_quant_name += operator_id + "_weight_" + std::to_string(layer_idx) + "_0:in";
  kernel_tensor->SetName(kernel_quant_name);

  std::vector<float> channel_scale(kernel_shape.dim[0]);
  std::vector<int> channel_zero_point(kernel_shape.dim[0]);
  std::vector<float> channel_max(kernel_shape.dim[0]);
  std::vector<float> channel_min(kernel_shape.dim[0]);

  const rapidjson::Value& scale = docWgt[(kernel_quant_name).c_str()]["scale"];
  const rapidjson::Value& zero_point = docWgt[(kernel_quant_name).c_str()]["zero_point"];
  const rapidjson::Value& max = docWgt[(kernel_quant_name).c_str()]["max_value"];
  const rapidjson::Value& min = docWgt[(kernel_quant_name).c_str()]["min_value"];

  if (scale.Size() > 1) {
    assert(scale.Size() == kernel_shape.dim[0]);
    for (int i = 0; i < kernel_shape.dim[0]; i++) {
      channel_scale[i] = scale[i].GetFloat();
      if (zero_point.IsArray()) {
        channel_zero_point[i] = zero_point[i].GetFloat();
      } else {
        channel_zero_point[i] = zero_point.GetFloat();
      }
      channel_max[i] = max[i].GetFloat();
      channel_min[i] = min[i].GetFloat();
    }

    kernel_tensor->SetChannelScale(channel_scale.data(), kernel_shape.dim[0]);
    kernel_tensor->SetChannelZeroPoint(channel_zero_point.data(), kernel_shape.dim[0]);
    kernel_tensor->SetChannelMax(channel_max.data(), kernel_shape.dim[0]);
    kernel_tensor->SetChannelMin(channel_min.data(), kernel_shape.dim[0]);
    if (std::strcmp(docWgt[(kernel_quant_name).c_str()]["quantizer"].GetString(), "Symmetric") == 0)
      kernel_tensor->SetQuantType(kQT_Sym);
    else
      kernel_tensor->SetQuantType(kQT_Asym);
  } else {
    kernel_tensor->SetScale(docWgt[(kernel_quant_name).c_str()]["scale"][0].GetFloat());

    kernel_tensor->SetMax(docWgt[(kernel_quant_name).c_str()]["max_value"][0].GetFloat());
    kernel_tensor->SetMin(docWgt[(kernel_quant_name).c_str()]["min_value"][0].GetFloat());
    // param.quant_mode = kQM_PerTensor;
    if (std::strcmp(docWgt[(kernel_quant_name).c_str()]["quantizer"].GetString(), "Symmetric") ==
        0) {
      kernel_tensor->SetQuantType(kQT_Sym);
      kernel_tensor->SetZeroPoint(docWgt[(kernel_quant_name).c_str()]["zero_point"].GetFloat());
    } else {
      kernel_tensor->SetQuantType(kQT_Asym);
      kernel_tensor->SetZeroPoint(docWgt[(kernel_quant_name).c_str()]["zero_point"][0].GetFloat());
    }
  }

  return outputs;
}

TensorVec VisVpuCodegen::BroadCastTo(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";  //
  auto input_name = inputs.at(0);

  TensorPtr input_tensor = graph->FindTensor(input_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, EXPAND_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetOutputTensor(0, output_tensor);

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  return outputs;
}

TensorVec VisVpuCodegen::Power(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 2) << "number of inputs need to corrected";  //
  auto input_name = inputs.at(0);
  // check exponent
  const auto* exponent = call->args[1].as<ConstantNode>();
  ICHECK(exponent);
  ICHECK_EQ(*static_cast<float*>(exponent->data->data), 2) << "Exponent must be 2";
  TensorPtr input_tensor = graph->FindTensor(input_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, SQUARE_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetOutputTensor(0, output_tensor);

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  return outputs;
}

TensorVec VisVpuCodegen::Squeeze(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";
  auto input_name = inputs.at(0);

  TensorPtr input_tensor = graph->FindTensor(input_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, RESHAPE_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetOutputTensor(0, output_tensor);

  // config param
  ReshapeOpPtr op_ptr = std::dynamic_pointer_cast<ReshapeOp>(node->GetOp());  //

  op_ptr->SetTargetShape(output_tensor->GetShape());

  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  return outputs;
}

TensorVec VisVpuCodegen::Take(const OpBuildArgs& op_build_args) {
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 2) << "number of inputs need to corrected";  //
  auto input_name = inputs.at(0);

  TensorPtr input_tensor = graph->FindTensor(input_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor_slice = graph->CreateTensor(op_name + "_slice:out", kTT_Var, out_type);
  TensorPtr output_tensor = graph->CreateTensor(op_name + "_reshape:out", kTT_Var, out_type);
  NodePtr slice_node = graph->CreateNode(op_name + "_slice", SLICE_OP);  //

  slice_node->SetInputTensor(0, input_tensor);
  slice_node->SetOutputTensor(0, output_tensor_slice);

  NodePtr reshape_node = graph->CreateNode(op_name + "_reshape", RESHAPE_OP);
  reshape_node->SetInputTensor(0, output_tensor_slice);
  reshape_node->SetOutputTensor(0, output_tensor);

  // parse attr
  const auto* attrs = call->attrs.as<TakeAttrs>();  //
  // get take axis
  int axis = int64_t(attrs->axis);
  // get indices
  const auto* indices = call->args[1].as<ConstantNode>();
  ICHECK(indices);
  int indices_val = *static_cast<int*>(indices->data->data);

  Array<Integer> axis_array;
  Array<Integer> begin_array;
  Array<Integer> end_array;
  Array<Integer> strides_array;

  axis_array.push_back(axis);
  begin_array.push_back(indices_val);
  end_array.push_back(indices_val + 1);
  strides_array.push_back(1);

  auto axes = GetConcrete_int<int>(axis_array);
  auto begin = GetConcrete_int<int>(begin_array);
  auto end = GetConcrete_int<int>(end_array);
  auto strides = GetConcrete_int<int>(strides_array);

  // config param
  SliceOpPtr op_ptr = std::dynamic_pointer_cast<SliceOp>(slice_node->GetOp());  //

  SliceParam& param = op_ptr->GetParam();  //

  for (size_t i = 0; i < axes.size(); i++) {
    param.axes.push_back(axes[i]);
    param.starts.push_back(begin[i]);
    param.ends.push_back(end[i]);
    param.steps.push_back(strides[i]);
  }

  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  // config reshepe param
  ReshapeOpPtr rs_op_ptr = std::dynamic_pointer_cast<ReshapeOp>(reshape_node->GetOp());  //
  rs_op_ptr->SetTargetShape(GetShapeType(oshape));

  // set shape
  ShapeType in_shape = input_tensor->GetShape();
  if (in_shape.dim_num == 3) {
    output_tensor_slice->SetShape(ShapeType(1, 1, oshape[1]));
  } else if (in_shape.dim_num == 4) {
    output_tensor_slice->SetShape(ShapeType(1, oshape[0], oshape[1], oshape[2]));
  } else {
    output_tensor_slice->SetShape(ShapeType(1, oshape[0], oshape[1], oshape[2], oshape[3]));
  }
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor_slice->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor_slice->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor_slice->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor_slice->SetQuantType(kQT_Sym);
    output_tensor_slice->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor_slice->SetQuantType(kQT_Asym);
    output_tensor_slice->SetZeroPoint(
        docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + "_reshape:out"};

  return outputs;
}

TensorVec VisVpuCodegen::Global_Sum(const OpBuildArgs& op_build_args) {  //
  UNPACK_OP_BUILD_ARGS(op_build_args);

  // create node
  CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";
  auto input_name = inputs.at(0);

  TensorPtr input_tensor = graph->FindTensor(input_name);

  vis::megrez::DataType out_type = Get_quant_type(op_name);
  TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

  NodePtr node = graph->CreateNode(op_name, GLOBAL_SUM_OP);  //

  node->SetInputTensor(0, input_tensor);
  node->SetOutputTensor(0, output_tensor);
  // set shape
  auto oshape = tvm::relay::backend::GetShape(call->checked_type());
  output_tensor->SetShape(GetShapeType(oshape));

  // quant output
  output_tensor->SetScale(docAct[(op_name + ":out").c_str()]["scale"][0].GetFloat());

  output_tensor->SetMax(docAct[(op_name + ":out").c_str()]["max_value"][0].GetFloat());
  output_tensor->SetMin(docAct[(op_name + ":out").c_str()]["min_value"][0].GetFloat());
  if (std::strcmp(docAct[(op_name + ":out").c_str()]["quantizer"].GetString(), "Symmetric") == 0) {
    output_tensor->SetQuantType(kQT_Sym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"].GetFloat());
  } else {
    output_tensor->SetQuantType(kQT_Asym);
    output_tensor->SetZeroPoint(docAct[(op_name + ":out").c_str()]["zero_point"][0].GetFloat());
  }

  decltype(inputs) outputs{op_name + ":out"};

  return outputs;
}

TensorVec VisVpuCodegen::Pass(const OpBuildArgs& op_build_args) {  //
  const std::vector<std::string>& inputs = op_build_args.inputs;

  // create node
  //  CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";//
  auto input_name = inputs.at(0);  // inputs.at(1) is quant param
  layer_idx = layer_idx - 1;

  decltype(inputs) outputs{input_name};
  return outputs;
}

TensorVec VisVpuCodegen::Build(const Call& call, const std::vector<std::string>& inputs) {
  auto op_node = call->op.as<OpNode>();
  CHECK(op_node);
  auto operator_id = op_node->name;

  auto build_op = op_map.find(operator_id);
  CHECK(build_op != op_map.end()) << "operator " << operator_id << " not supported yet.";

  // auto build_func = this->*build_op->second;
  // get name
  std::string op_name;

  op_name += operator_id + "_" + std::to_string(layer_idx);
  // get output name

  LOG(INFO) << "build VisVpu op: " << op_name;

  OpBuildArgs args{op_name, call, inputs};
  auto res = (this->*build_op->second)(args);
  layer_idx = layer_idx + 1;
  return res;
}

vis::megrez::DataType VisVpuCodegen::Get_quant_type(const std::string& op_name) {
  int bit_width = docAct[(op_name + ":out").c_str()]["bit_width"].GetInt();

  std::string quantizer = docAct[(op_name + ":out").c_str()]["quantizer"].GetString();

  vis::megrez::DataType out_type = kDT_Int8;
  if (bit_width == 8) {
    if (quantizer == "Symmetric") {
      out_type = kDT_Int8;
    } else if (quantizer == "Asymmetric") {
      out_type = kDT_Uint8;
    } else {
      LOG(FATAL) << "Unknown DataType";
    }
  } else if (bit_width == 16) {
    if (quantizer == "Symmetric") {
      out_type = kDT_Int16;
    } else if (quantizer == "Asymmetric") {
      out_type = kDT_Uint16;
    } else {
      LOG(FATAL) << "Unknown DataType";
    }
  } else if (bit_width == 32) {
    out_type = kDT_Int32;
  } else {
    LOG(FATAL) << "Unknown DataType";
  }

  return out_type;
}

vis::megrez::DataType VisVpuCodegen::Get_split_type(const std::string& op_name, const int index) {
  int bit_width = docAct[(op_name + ":out_" + std::to_string(index)).c_str()]["bit_width"].GetInt();
  std::string quantizer =
      docAct[(op_name + ":out_" + std::to_string(index)).c_str()]["quantizer"].GetString();
  vis::megrez::DataType out_type = kDT_Int8;
  if (bit_width == 8) {
    if (quantizer == "Symmetric") {
      out_type = kDT_Int8;
    } else if (quantizer == "Asymmetric") {
      out_type = kDT_Uint8;
    } else {
      LOG(FATAL) << "Unknown DataType";
    }
  } else if (bit_width == 16) {
    if (quantizer == "Symmetric") {
      out_type = kDT_Int16;
    } else if (quantizer == "Asymmetric") {
      out_type = kDT_Uint16;
    } else {
      LOG(FATAL) << "Unknown DataType";
    }
  } else if (bit_width == 32) {
    out_type = kDT_Int32;
  } else {
    LOG(FATAL) << "Unknown DataType";
  }

  return out_type;
}
/*! \breief Convert TVM DataType to VISVPU's. */

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
