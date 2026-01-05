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
 * \file src/relay/backend/contrib/vis/codegen.cc
 * \brief Implementation of vis Relay codegen.
 */


#include <iostream>
#include "codegen_vis_vpu.h"

#include <tvm/relay/attrs/nn.h>

using namespace vis::megrez;

namespace tvm {
namespace relay {
namespace contrib {

VisVpuCodegen::VisVpuCodegen(std::string symbol, Expr expr)
    : MixedModeVisitor(1), symbol_(std::move(symbol)), func_(std::move(expr)) {
    auto ctx = transform::PassContext::Current();

    auto cfg = ctx->GetConfig<VisVpuConfig>("relay.ext.vis_vpu.options");
    if (!cfg.defined()) {
        cfg = AttrsWithDefaultValues<VisVpuConfig>();
    }
    config_ = cfg.value();
}



void VisVpuCodegen::Init() {
    op_map["nn.conv2d"] = &VisVpuCodegen::Conv2D;
    op_map["add"] = &VisVpuCodegen::ElementWise_add;
    op_map["divide"] = &VisVpuCodegen::ElementWise_div;
    op_map["nn.avg_pool2d"] = &VisVpuCodegen::Pool2D_avg;
    op_map["clip"] = &VisVpuCodegen::Clip;
    op_map["concatenate"] = &VisVpuCodegen::Concatenate;
    op_map["nn.conv2d_transpose"] = &VisVpuCodegen::Conv2D_transpose;
    op_map["nn.dense"] = &VisVpuCodegen::Dense;
    op_map["nn.batch_flatten"] = &VisVpuCodegen::Flatten;
    op_map["nn.global_avg_pool2d"] = &VisVpuCodegen::Global_Pool2D_avg;
    op_map["nn.global_max_pool2d"] = &VisVpuCodegen::Global_Pool2D_max;
    op_map["nn.adaptive_avg_pool1d"] = &VisVpuCodegen::Global_Pool2D_avg;
    op_map["nn.leaky_relu"] = &VisVpuCodegen::LeakyReLU;
    op_map["nn.max_pool2d"] = &VisVpuCodegen::Pool2D_max;
    op_map["multiply"] = &VisVpuCodegen::Mul;
    op_map["mean"] = &VisVpuCodegen::Mean;
    op_map["nn.relu"] = &VisVpuCodegen::Relu;
    op_map["reshape"] = &VisVpuCodegen::Reshape;
    op_map["image.resize2d"] = &VisVpuCodegen::Resize;
    op_map["sigmoid"] = &VisVpuCodegen::Sigmoid;
    op_map["strided_slice"] = &VisVpuCodegen::Slice;
    op_map["split"] = &VisVpuCodegen::Split;
    op_map["transpose"] = &VisVpuCodegen::Transpose;
	op_map["nn.upsampling"] = &VisVpuCodegen::Upsample;
    op_map["nn.batch_norm"] = &VisVpuCodegen::Batch_norm;
    op_map["nn.pad"] = &VisVpuCodegen::Pad;
    op_map["subtract"] = &VisVpuCodegen::ElementWise_sub;
    op_map["sqrt"] = &VisVpuCodegen::Sqrt;
    op_map["nn.batch_matmul"] = &VisVpuCodegen::Bmatmul;
    op_map["nn.softmax"] = &VisVpuCodegen::Softmax;
    op_map["expand_dims"] = &VisVpuCodegen::Unsqueeze;
    op_map["broadcast_to"] = &VisVpuCodegen::BroadCastTo;
    op_map["power"] = &VisVpuCodegen::Power;
    op_map["squeeze"] = &VisVpuCodegen::Squeeze;
    op_map["take"] = &VisVpuCodegen::Take;
    op_map["sum"] = &VisVpuCodegen::Global_Sum;
    op_map["exp"] = &VisVpuCodegen::Exp;
    op_map["tanh"] = &VisVpuCodegen::Tanh;


    op_map["annotation.stop_fusion"] = &VisVpuCodegen::Pass;
    op_map["relay.op.annotation.simulated_quantize"] = &VisVpuCodegen::Pass;
    op_map["annotation.cast_hint"] = &VisVpuCodegen::Pass;


    graph = NULL;

    graph = Graph::Create(symbol_);

    const_idx = 0;

    layer_idx = 0;
}

void VisVpuCodegen::BuildGraph() {


    Init();

    std::string calibTableFileAct = /*i->calibTablesPath + "/" + */config_->calibtableact;
    std::string calibTableFileWgt = /*i->calibTablesPath + "/" + */config_->calibtablewgt;
    FILE* fp1 = fopen(calibTableFileAct.c_str(), "r");
    FILE* fp2 = fopen(calibTableFileWgt.c_str(), "r");
    CHECK(fp1 != NULL && fp2 != NULL)<<"can not open file.";
    char readBuffer1[TEST_PARAM_FILE_MAX_SIZE] = {0};
    char readBuffer2[TEST_PARAM_FILE_MAX_SIZE] = {0};
    rapidjson::FileReadStream inStrAct(fp1, readBuffer1, sizeof(readBuffer1));
    rapidjson::FileReadStream inStrWgt(fp2, readBuffer2, sizeof(readBuffer2));

    docAct.ParseStream(inStrAct);
    docWgt.ParseStream(inStrWgt);
    
    if (config_->tilingJson != "none")
    {
        std::string tilingJsonPath = /*i->calibTablesPath + "/" + */config_->tilingJson;
        FILE* fp3 = fopen(tilingJsonPath.c_str(), "r");
        CHECK(fp3 != NULL)<<"can not open file.";
        char readBuffer3[TEST_TILING_FILE_MAX_SIZE] = {0};

        rapidjson::FileReadStream inStrTiling(fp3, readBuffer3, sizeof(readBuffer3));

        docTiling.ParseStream(inStrTiling);
        doTiling = true;

        useTiling =  docTiling["network"]["net_need_fusion"].GetBool();
    }
    // mem_allocator
    if (config_->allocateJson != "none" && config_->tilingJson != "none")
    {
        doAllocate = true;
        // in visinex npu, only when doTiling and doAllocate and dont_have_transpose, can use output of this allocator
        useAllocate = doTiling && doAllocate && useTiling;
        // useAllocate = false;
        mem_allocator_builder_ = MemoryAllocatorBuilderPtr(new MemoryAllocatorBuilder(config_->memPoolSize, config_->memPoolSize/64, config_->allocateJson, config_->dataReuse, config_->opAddrReuse));
    }

    auto func = Downcast<relay::Function>(func_);

    // Build input tensors
    for (size_t idx = 0; idx < func->params.size(); ++idx) {
        const auto& param = func->params[idx];
        const auto& tn = param->checked_type().as<TensorTypeNode>();
        CHECK(tn) << "not tensor type";
        //create input
        std::string input_name = "network_input_" + std::to_string(idx);
        // auto dtype = tn->dtype;
        //TensorPtr input_tensor = graph->CreateTensor(input_name, kTT_Input, Tvm2VisVpu(dtype));
	auto dataType = kDT_Int8;
        if (docAct[input_name.c_str()]["bit_width"].GetInt() == 16)
        {
            dataType = kDT_Int16;
        }

        TensorPtr input_tensor = graph->CreateTensor(input_name, kTT_Input, dataType);
        LOG(INFO) << "(VisVpu) mark input:" << input_tensor->GetName() << " dataType=" << dataType << std::endl;

        //shape
        auto shape = GetConcrete<int>(tn->shape);
        input_tensor->SetShape(GetShapeType(shape));

        //quant
        input_tensor->SetScale(docAct[input_name.c_str()]["scale"][0].GetFloat());
        input_tensor->SetMax(docAct[input_name.c_str()]["max_value"][0].GetFloat());
        input_tensor->SetMin(docAct[input_name.c_str()]["min_value"][0].GetFloat());
		if  (std::strcmp(docAct[input_name.c_str()]["quantizer"].GetString(),"Symmetric")==0){
			input_tensor->SetQuantType(kQT_Sym);
			input_tensor->SetZeroPoint(docAct[input_name.c_str()]["zero_point"].GetFloat());}
		else {
			input_tensor->SetQuantType(kQT_Asym);
			input_tensor->SetZeroPoint(docAct[input_name.c_str()]["zero_point"][0].GetFloat());}

        CHECK(vis_tensor_map_.find(param) == vis_tensor_map_.end());
        //Tensor list
        vis_tensor_map_[param].emplace_back(input_name);
    }

    // Build ops and internal tensors
    VisitExpr(func->body);

    // Mark output
    CHECK(vis_tensor_map_.find(func->body) != vis_tensor_map_.end());  // .at() gives bad message
    const auto& outputs = vis_tensor_map_[func->body];

    for (size_t idx = 0; idx < outputs.size(); ++idx) {
        const auto& name = outputs[idx];
        TensorPtr output_tensor = graph->FindTensor(name);
        output_tensor->SetTensorType(kTT_Output);
        LOG(INFO) << "(VisVpu) mark ouput:" << output_tensor->GetName() << std::endl;
    }

    //---------------------------------------------------------//
    //deal with graph
    std::cout << "before finalize\n";

    graph->Finalize();
    std::cout << "after finalize\n";
    graph->Save(std::cout);

    auto npu_dev = vis::megrez::NPUDevice::Create("megrez");
    std::cout << "npu_dev created\n";
    if (npu_dev == nullptr) std::cerr << "failed to create npu device\n";

    //npu_dev->SetMemPoolSize(config_->memPoolSize);//todo param

    std::cout << npu_dev->ToString() << "\n";

    auto builder = ModelBuilder::Create(graph, npu_dev); //todo

    ParamSet option;
    int force_free_level=FORCE_FREE_ALL;

    const char * str_free_level=std::getenv("FORCE_FREE_LEVEL");

    if(str_free_level) {
	force_free_level=std::atoi(str_free_level);

	if(force_free_level<0 || force_free_level>FORCE_FREE_ALL)
	{
           std::cerr<<"bad force free level env var: "<<str_free_level<<"\n";
	   return;
	}
    }

    option["force_free"]=force_free_level;

    std::cout<<"FORCE_FREE_LEVEL: "<<force_free_level<<"\n";


    std::vector<std::string> ddr_tensors;

#if 0
    /// add the tensor here
     /// init list
    std::vector<NodePtr> nodes;
    nodes = graph->GetNodeList();
    std::cout << "exlcusive_tensor:\n";
    for (auto& n : nodes)
    {
	    std::cout << n->GetName();
	    ddr_tensors.push_back(n->GetName()+":out");

    }
    std::cout << "\nexclusive_tensor end.\n";

    std::ofstream nodes_stream;

    nodes_stream.open("nodes.yaml");

    graph->SaveNodesMap(nodes_stream);
    //ddr_tensors.push_back("nn.conv2d_3:out");
    //ddr_tensors.push_back("add_41:out");
#endif

    option["exclusive_tensor"]=ddr_tensors;

    if (doTiling)
    {
        printf("start_do_tiling!!!!\n");
        // int group_num = docTiling["network"]["group_num"].GetInt();
        int group_num = tile_sort_map.size();
        // assert(group_num == tile_sort_map.size());
        bool flag = true;
        for (int i = 0; i < group_num; i++)
        {
            GroupInfoPtr group_info = tile_sort_map[i];
            if (!group_info->need_tiling)
            {
                break;
            }
            // if (group_info->dont_support)
            // {
            //     break;
            // }
            int call_num = group_info->call_index_list.size();
            if (flag)
            {
                if (group_info->pre_process.size() > 0)
                {
                    tiling_tile_stream.push_back(group_info->pre_process[0] + ".tile0");
                    printf("%s\n", (group_info->pre_process[0] + ".tile0").c_str());
                }
                for (int j = 0; j < group_info->tile_num; j++)
                {
                    for (int k = 0; k < call_num; k++)
                    {
                        std::string tile_name = group_info->call_name_list[k] + ".tile" + std::to_string(j);
                        tiling_tile_stream.push_back(tile_name);
                        printf("%s\n", tile_name.c_str());
                    }
                }
                if (group_info->post_process.size() > 0)
                {
                    tiling_tile_stream.push_back(group_info->post_process[0] + ".tile0");
                    printf("%s\n", (group_info->post_process[0] + ".tile0").c_str());
                }
                
                if(config_->newSchedule) 
                {
                    flag = false;
                }
            }
            else{
                if (group_info->pre_process.size() > 0)
                {
                    tiling_tile_stream.push_back(group_info->pre_process[0] + ".tile0");
                    printf("%s\n", (group_info->pre_process[0] + ".tile0").c_str());
                }
                for (int j = group_info->tile_num - 1; j >=0 ; j--)
                {
                    for (int k = 0; k < call_num; k++)
                    {
                        std::string tile_name = group_info->call_name_list[k] + ".tile" + std::to_string(j);
                        tiling_tile_stream.push_back(tile_name);
                        printf("%s\n", tile_name.c_str());
                    }
                }
                if (group_info->post_process.size() > 0)
                {
                    tiling_tile_stream.push_back(group_info->post_process[0] + ".tile0");
                    printf("%s\n", (group_info->post_process[0] + ".tile0").c_str());
                }
                flag = true;
            }
            
        }

        std::string first_name = tile_sort_map[0]->call_name_list[0];
        if (useTiling)
        {
            std::fstream file;
            file.open("./tiling_tile_stream.txt",std::ios::out); //以只写模式打开文件
            for(int i = 0; i < int(tiling_tile_stream.size()); i++)
            {
                // std::string tile_name = tiling_tile_stream[i];
                // char charArray[tile_name.size() + 1];
                // strcpy(charArray, tile_name.c_str());
                // file.write(charArray,sizeof(charArray));//写入文件
                file << tiling_tile_stream[i] << std::endl;
            }
            file.close();

            tile_sched[first_name] = tiling_tile_stream;
            option["tile_sched"] = &tile_sched;
        }
    }

    if (doAllocate)
    {
        int group_num = tile_sort_map.size();
        // assert(group_num == tile_sort_map.size());
        bool flag = true;
        for (int i = 0; i < group_num; i++)
        {
            GroupInfoPtr group_info = tile_sort_map[i];
            int call_num = group_info->call_index_list.size();
            if (flag)
            {
                if (group_info->pre_process.size() > 0)
                {
                    allo_tile_stream.push_back(group_info->pre_process[0] + ".tile0");
                    printf("%s\n", (group_info->pre_process[0] + ".tile0").c_str());
                }
                for (int j = 0; j < group_info->tile_num; j++)
                {
                    for (int k = 0; k < call_num; k++)
                    {
                        std::string tile_name = group_info->call_name_list[k] + ".tile" + std::to_string(j);
                        allo_tile_stream.push_back(tile_name);
                        printf("%s\n", tile_name.c_str());
                    }
                }
                if (group_info->post_process.size() > 0)
                {
                    allo_tile_stream.push_back(group_info->post_process[0] + ".tile0");
                    printf("%s\n", (group_info->post_process[0] + ".tile0").c_str());
                }

                if(config_->newSchedule) 
                {
                    flag = false;
                }
            }
            else{
                if (group_info->pre_process.size() > 0)
                {
                    allo_tile_stream.push_back(group_info->pre_process[0] + ".tile0");
                    printf("%s\n", (group_info->pre_process[0] + ".tile0").c_str());
                }
                for (int j = group_info->tile_num - 1; j >=0 ; j--)
                {
                    for (int k = 0; k < call_num; k++)
                    {
                        std::string tile_name = group_info->call_name_list[k] + ".tile" + std::to_string(j);
                        allo_tile_stream.push_back(tile_name);
                        printf("%s\n", tile_name.c_str());
                    }
                }
                if (group_info->post_process.size() > 0)
                {
                    allo_tile_stream.push_back(group_info->post_process[0] + ".tile0");
                    printf("%s\n", (group_info->post_process[0] + ".tile0").c_str());
                }
                flag = true;
            }  
        }
        for (size_t i = 0; i < allo_tile_stream.size(); i++)
        {
            mem_allocator_builder_->SetEpoch(allo_tile_stream[i], i);
        }

        std::fstream file;
        file.open("./allo_tile_stream.txt",std::ios::out); //以只写模式打开文件
        for(int i = 0; i < int(allo_tile_stream.size()); i++)
        {
            // std::string tile_name = allo_tile_stream[i];
            // char charArray[tile_name.size() + 1];
            // strcpy(charArray, tile_name.c_str());
            // file.write(charArray,sizeof(charArray));//写入文件
            file << allo_tile_stream[i] << std::endl;
        }
        file.close();
        printf("***********DoMemAlocator************\n");
        mem_allocator_builder_->DoMemAlocator();
    }
    
    if (useAllocate)
    {
        printf("***********UseAllocator************\n");
        option["mem_allo_json"]=config_->allocateJson;
    }

    bool ret = builder->Compile(option);//todo

    if (!ret)
    {
        std::cout << "build failed\n";
        return;
    }

    std::cout << "build done\n";

    std::cout << "check build result:\n";

    std::cout << "mem map:\n";
    MemMap mem_map;
    builder->GetMemMap(mem_map);

    std::cout << MemMapToString(mem_map) << "\n";
    if (ddr_tensors.size() > 0)
    {
        std::vector<TensorMapItem> tensor_list;
       	builder->GetFeatureMapTensorMap(tensor_list);
        SaveMemMap("ddr.map", mem_map, tensor_list);
    }
    else
    {
        SaveMemMap("ddr.map", mem_map);
    }

    std::cout << "assembly code:\n";

    InstStream ins_stream;

    builder->GetInstStream(ins_stream);

    std::ofstream insfile_stream;

    insfile_stream.open(config_->insFilename);

    ins_stream.Save(insfile_stream, false, true); //todo param insfilename

    std::cout << "instruction mapping:\n";

    InstMap inst_map;

    builder->GetInstMap(inst_map);

    inst_map.Save(std::cout);

    std::cout << "get param:\n";

    ParamBuffer param_buf;

    builder->GetDDRParam(param_buf);

    std::cout << "param size: " << param_buf.data_size << "\n";

    std::ofstream param_stream;

    param_stream.open(config_->paramFilename);

    if (param_stream.is_open()) {
        param_stream.write((char*)param_buf.data, param_buf.data_size);
    }

    //---------------------------------------------------------//
    fclose(fp1);
    fclose(fp2);
}

void VisVpuCodegen::VisitExpr_(const ConstantNode* cn) {
    auto cnst = GetRef<Constant>(cn);
    printf("Into Constant Node.\n");
    CHECK(vis_tensor_map_.find(cnst) == vis_tensor_map_.end());

    std::string name = "const" + std::to_string(const_idx);//temp name
    const_idx = const_idx + 1;
    auto tt = cn->tensor_type();
    // auto dtype = tt->dtype;
    // CHECK(dtype == DataType::Float(32));
    auto shape = GetConcrete<int>(tt->shape);

    auto data_container = cn->data.operator->();
    auto data_ptr = static_cast<uint8_t*>(data_container->data) + data_container->byte_offset;
    auto data_type = cn->data.DataType();

    // std::cout<<" data_type->==::"<< data_type<<std::endl;
    // CHECK(data_type == DataType::Float(32));
    TensorPtr weight_tensor = graph->CreateTensor(name, kTT_Const, Tvm2VisVpu(data_type));
    //TensorPtr weight_tensor = graph->CreateTensor(name, kTT_Const, kDT_Int8);
    weight_tensor->SetShape(GetShapeType(shape));

    /// allocate memory
    int mem_size = weight_tensor->GetMemSize();

    weight_tensor->SetData(data_ptr, mem_size);

    //std::free(data_ptr);

    vis_tensor_map_[cnst] = {name};

    LOG(INFO) << "add vis vpu tensor const: " << name ;

}

void VisVpuCodegen::VisitExpr_(const FunctionNode* cn) {
}

void VisVpuCodegen::VisitExpr_(const CallNode* cn) {
    auto call = GetRef<Call>(cn);
    CHECK(vis_tensor_map_.find(call) == vis_tensor_map_.end());

    //dealing with pattern
    if (const auto* func = cn->op.as<FunctionNode>()){
        using backend::GetRootCall;

        printf("Into Function Node.\n");
        const auto pattern_name = func->GetAttr<runtime::String>(attr::kComposite);
        ICHECK(pattern_name.defined()) << "Only functions with composite attribute supported";
        if(pattern_name == "vis_vpu.conv2d_bias")
        {
            // build input tensors
            printf("Into vis_vpu.conv2d_bias Node.\n");
            bool with_relu = false;
            std::vector<std::string> inputs;
            for (const auto& arg : cn->args) {
                CHECK(vis_tensor_map_.find(arg) != vis_tensor_map_.end());  // .at() gives bad message
                const auto& ts = vis_tensor_map_[arg];
                CHECK_EQ(ts.size(), 1) << "call arg is not expected be tuple";
                inputs.emplace_back(ts.back());
            }

            // Traverse composite convolution function from child to parent
            std::vector<std::string> op_namelist = {"nn.conv2d", "add"};
            std::vector<std::string> op_namelist_1 = {"add"};
            int add_base = 0;
            const auto* last_call = func->body.as<CallNode>();

            if (backend::IsOp(last_call, "nn.relu")) {
                printf("conv2d_bias_relu.\n");
                with_relu = true;
                op_namelist.push_back("nn.relu");
                op_namelist_1.push_back("nn.relu");
                add_base = 1;
            }
            printf("conv_call.\n");
            const CallNode* conv_call = GetRootCall(func->body.as<CallNode>(), add_base + 1, op_namelist);
            printf("add_call.\n");
            const CallNode* add_call = GetRootCall(func->body.as<CallNode>(), add_base + 0, op_namelist_1);
            printf("111.\n");
            // set bias tensor data
            auto cnst = GetRef<Constant>(add_call->args[1].as<ConstantNode>());
            CHECK(vis_tensor_map_.find(cnst) == vis_tensor_map_.end());
            std::string name = "bias" + std::to_string(layer_idx + 1);  // temp name

            auto tt = add_call->args[1].as<ConstantNode>()->tensor_type();
            auto shape = GetConcrete<int>(tt->shape);

            auto data_container = add_call->args[1].as<ConstantNode>()->data.operator->();
            auto data_ptr = static_cast<uint8_t*>(data_container->data) + data_container->byte_offset;
            auto data_type = add_call->args[1].as<ConstantNode>()->data.DataType();

            TensorPtr bias_tensor = graph->CreateTensor(name, kTT_Const, Tvm2VisVpu(data_type));
            auto bias_shape = GetShapeType(shape);
            bias_tensor->SetShape(bias_shape);
            /// allocate memory
            int mem_size = bias_tensor->GetMemSize();
            bias_tensor->SetData(data_ptr, mem_size);

            //create name

            std::string op_name = "nn.conv2d_" + std::to_string(layer_idx);
            std::string op_name_last = "add_" + std::to_string(layer_idx+1);
            // if (with_relu)
            // {
            //     op_name_last = "relu_" + std::to_string(layer_idx+2);
            // }
			std::cout<<"op_name = "<<op_name<<std::endl;

            //create node

            CHECK_EQ(inputs.size(), 2) << "number of inputs need to corrected";
            auto input_name = inputs.at(0);
            auto kernel_name = inputs.at(1);

            TensorPtr input_tensor = graph->FindTensor(input_name);
            TensorPtr kernel_tensor = graph->FindTensor(kernel_name);

            vis::megrez::DataType out_type = Get_quant_type(op_name_last);

            TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

            NodePtr node = graph->CreateNode(op_name, CONV_OP);//

            node->SetInputTensor(0, input_tensor);
            node->SetInputTensor(1, kernel_tensor);
            node->SetInputTensor(2, bias_tensor);
            node->SetOutputTensor(0, output_tensor);

            auto kernel_shape = kernel_tensor->GetShape();

            //parse attr
            auto first_call = GetRef<Call>(conv_call);
            const auto* attrs = first_call->attrs.as<Conv2DAttrs>();//

            CHECK(attrs) << "not Conv2DAttrs node";//

            CHECK_EQ(attrs->data_layout, "NCHW") << "layout " << attrs->data_layout << " not supported";//

            auto strides = GetConcrete<int>(attrs->strides);
            auto padding = GetConcrete<int>(attrs->padding);
            auto dilations = GetConcrete<int>(attrs->dilation);
            auto kernel_size = GetConcrete<int>(attrs->kernel_size);

            int groups = attrs->groups;

            // config param
            std::cout<<"config param = "<<op_name<<std::endl;
            ConvOpPtr op_ptr = std::dynamic_pointer_cast<ConvOp>(node->GetOp());//

            ConvParam& param = op_ptr->GetParam();//

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

            if (with_relu)
            {
                param.activation = 1;
            }

            //set shape
            // auto last_call = GetRef<Call>(add_call);
            auto oshape = tvm::relay::backend::GetShape(last_call->checked_type());
            printf("[%d,%d,%d,%d]\n", oshape[0],oshape[1],oshape[2],oshape[3]);
            output_tensor->SetShape(GetShapeType(oshape));

            //quant output

            output_tensor->SetScale(docAct[(op_name_last + ":out").c_str()]["scale"][0].GetFloat());
            output_tensor->SetMax(docAct[(op_name_last + ":out").c_str()]["max_value"][0].GetFloat());
            output_tensor->SetMin(docAct[(op_name_last + ":out").c_str()]["min_value"][0].GetFloat());
			if  (std::strcmp(docAct[(op_name_last + ":out").c_str()]["quantizer"].GetString(),"Symmetric")==0) {
				output_tensor->SetQuantType(kQT_Sym);
				output_tensor->SetZeroPoint(docAct[(op_name_last + ":out").c_str()]["zero_point"].GetFloat());}

		    else {
				output_tensor->SetQuantType(kQT_Asym);
				output_tensor->SetZeroPoint(docAct[(op_name_last + ":out").c_str()]["zero_point"][0].GetFloat());}

            decltype(inputs) outputs{op_name + ":out"};

            //quant weight

            std::string kernel_quant_name;
            kernel_quant_name = kernel_quant_name + "nn.conv2d" + "_weight_" + std::to_string(layer_idx) + "_0:in";
            kernel_tensor->SetName(kernel_quant_name);
            std::string bias_quant_name;
            bias_quant_name = bias_quant_name + "add" + "_bias_" + std::to_string(layer_idx+1) + "_0:in";
            bias_tensor->SetName(bias_quant_name);

            std::vector<float> channel_scale(kernel_shape.dim[0]);
            std::vector<int> channel_zero_point(kernel_shape.dim[0]);
            std::vector<float> channel_max(kernel_shape.dim[0]);
            std::vector<float> channel_min(kernel_shape.dim[0]);

            const rapidjson::Value& scale = docWgt[(kernel_quant_name).c_str()]["scale"];
            const rapidjson::Value& zero_point = docWgt[(kernel_quant_name).c_str()]["zero_point"];
            const rapidjson::Value& max = docWgt[(kernel_quant_name).c_str()]["max_value"];
            const rapidjson::Value& min = docWgt[(kernel_quant_name).c_str()]["min_value"];
            if (scale.Size()>1) {

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
                printf("set perchannel!!!!\n");
                param.quant_mode = kQM_PerChannel;
				if  (std::strcmp(docWgt[(kernel_quant_name).c_str()]["quantizer"].GetString(),"Symmetric")==0)
					kernel_tensor->SetQuantType(kQT_Sym);
		    	else
					kernel_tensor->SetQuantType(kQT_Asym);
            }
				else {
            	kernel_tensor->SetScale(docWgt[(kernel_quant_name).c_str()]["scale"][0].GetFloat());
				kernel_tensor->SetMax(docWgt[(kernel_quant_name).c_str()]["max_value"][0].GetFloat());
				kernel_tensor->SetMin(docWgt[(kernel_quant_name).c_str()]["min_value"][0].GetFloat());
                param.quant_mode = kQM_PerTensor;
				if  (std::strcmp(docWgt[(kernel_quant_name).c_str()]["quantizer"].GetString(),"Symmetric")==0){
					kernel_tensor->SetQuantType(kQT_Sym);
					kernel_tensor->SetZeroPoint(docWgt[(kernel_quant_name).c_str()]["zero_point"].GetFloat());}
		    	else{
					kernel_tensor->SetQuantType(kQT_Asym);
					kernel_tensor->SetZeroPoint(docWgt[(kernel_quant_name).c_str()]["zero_point"][0].GetFloat());}
            }

            if (doTiling)
            {
                std::string tiling_name;
                tiling_name = op_name_last;
                int output_tile_num = docTiling[(tiling_name).c_str()]["output_tile_num"].GetInt();
                int output_dim_num = docTiling[(tiling_name).c_str()]["output_dim_num"].GetInt();
                assert(output_dim_num == 4);
                int output_tile_dim = docTiling[(tiling_name).c_str()]["output_tile_dims"][0].GetInt();
                std::vector<TileTensorShape> tile_shape_list;
                printf("***********conv************\n");
                printf("tiling_name: %s \n", (tiling_name).c_str());
                printf("tilenum: %d \n", output_tile_num);
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
                    std::string op_bias_name = bias_tensor->GetName();
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

                        if (i==0)
                        {
                            kernelTileShape[i] = ((kernelTileShape[i] % 4) == 0) ? kernelTileShape[i] : (kernelTileShape[i] / 4 + 1) * 4;
                        }
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
                    int index = docTiling[(op_name_last).c_str()]["weight_tile_pos"].Size() == 1 ? 0 : k;
                    int param_align = 352;
                    if (scale.Size() > 1) 
                    {
                        param_align = 544;
                    }
                    biasTilePos.push_back(0);
                    biasTileShape.push_back(docTiling[(op_name_last).c_str()]["weight_tile_pos"][index][0][1].GetInt() - docTiling[(op_name_last).c_str()]["weight_tile_pos"][index][0][0].GetInt());
                    biasTileShape[0] = ((biasTileShape[0] % 64) == 0) ? (biasTileShape[0] / 64) * param_align : (biasTileShape[0] / 64 + 1) * param_align;
                    op_bias_name = op_bias_name + ":" + std::to_string(docTiling[(op_name_last).c_str()]["weight_tile_pos"][index][0][0].GetInt());
                    op_bias_name = op_bias_name + ":" + std::to_string(docTiling[(op_name_last).c_str()]["weight_tile_pos"][index][0][1].GetInt());
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
                    if (layer_idx + 2 + add_base == docTiling["network"]["call_num"].GetInt())
                    {
                        output_block_type = BLOCK_TYPE_NET_OUTPUT;
                    }
                    printf("***********CreateBlock************\n");
                    mem_allocator_builder_->CreateBlock(tile_name, op_output_name, 
                        tile_output_name, outputTileShape, outputTilePos,
                        output_bit_width/8, output_block_type);

                    printf("***********************\n");

                }
            }

            //updata layer_idx for pattern
            layer_idx = layer_idx + 2 + add_base;

            //CHECK_EQ(inputs.size(), func->num_inputs) << "wrong input arity: " << inputs.size();
            vis_tensor_map_[call] = outputs;

        }
        else if(pattern_name == "vis_vpu.conv2d_bias_expdims"){
            // build input tensors
            printf("Into vis_vpu.conv2d_bias_expdims Node.\n");
            std::vector<std::string> inputs;
            for (const auto& arg : cn->args) {
                CHECK(vis_tensor_map_.find(arg) != vis_tensor_map_.end());  // .at() gives bad message
                const auto& ts = vis_tensor_map_[arg];
                CHECK_EQ(ts.size(), 1) << "call arg is not expected be tuple";
                inputs.emplace_back(ts.back());
            }

            std::vector<std::string> op_namelist = {"nn.conv2d", "expand_dims", "add"};
            const CallNode* conv_call = GetRootCall(func->body.as<CallNode>(), 2, op_namelist);
            std::vector<std::string> op_namelist_1 = {"add"};
            const CallNode* add_call = GetRootCall(func->body.as<CallNode>(), 0, op_namelist_1);

            // set bias tensor data
            auto cnst = GetRef<Constant>(add_call->args[1].as<ConstantNode>());
            CHECK(vis_tensor_map_.find(cnst) == vis_tensor_map_.end());
            std::string name = "bias" + std::to_string(layer_idx + 1);  // temp name

            auto tt = add_call->args[1].as<ConstantNode>()->tensor_type();
            auto shape = GetConcrete<int>(tt->shape);

            auto data_container = add_call->args[1].as<ConstantNode>()->data.operator->();
            auto data_ptr = static_cast<uint8_t*>(data_container->data) + data_container->byte_offset;
            auto data_type = add_call->args[1].as<ConstantNode>()->data.DataType();

            TensorPtr bias_tensor = graph->CreateTensor(name, kTT_Const, Tvm2VisVpu(data_type));
            auto bias_shape = GetShapeType(shape);
            bias_tensor->SetShape(bias_shape);
            /// allocate memory
            int mem_size = bias_tensor->GetMemSize();
            bias_tensor->SetData(data_ptr, mem_size);

            //create name

            std::string op_name = "nn.conv2d_" + std::to_string(layer_idx);
            std::string op_name_last = "add_" + std::to_string(layer_idx+2);

            //create node

            CHECK_EQ(inputs.size(), 2) << "number of inputs need to corrected";
            auto input_name = inputs.at(0);
            auto kernel_name = inputs.at(1);

            TensorPtr input_tensor = graph->FindTensor(input_name);
            TensorPtr kernel_tensor = graph->FindTensor(kernel_name);

            vis::megrez::DataType out_type = Get_quant_type(op_name_last);

            TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

            NodePtr node = graph->CreateNode(op_name, CONV_OP);//

            node->SetInputTensor(0, input_tensor);
            node->SetInputTensor(1, kernel_tensor);
            node->SetInputTensor(2, bias_tensor);
            node->SetOutputTensor(0, output_tensor);

            auto kernel_shape = kernel_tensor->GetShape();

            //parse attr
            auto first_call = GetRef<Call>(conv_call);
            const auto* attrs = first_call->attrs.as<Conv2DAttrs>();//

            CHECK(attrs) << "not Conv2DAttrs node";//

            CHECK_EQ(attrs->data_layout, "NCHW") << "layout " << attrs->data_layout << " not supported";//

            auto strides = GetConcrete<int>(attrs->strides);
            auto padding = GetConcrete<int>(attrs->padding);
            auto dilations = GetConcrete<int>(attrs->dilation);
            auto kernel_size = GetConcrete<int>(attrs->kernel_size);

            int groups = attrs->groups;

            // config param

            ConvOpPtr op_ptr = std::dynamic_pointer_cast<ConvOp>(node->GetOp());//

            ConvParam& param = op_ptr->GetParam();//

            param.pad_h0 = padding[0];
            param.pad_h1 = padding[1];
            param.pad_w0 = padding[2];
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

            //set shape
            auto last_call = GetRef<Call>(add_call);
            auto oshape = tvm::relay::backend::GetShape(last_call->checked_type());
            printf("[%d,%d,%d,%d]\n", oshape[0],oshape[1],oshape[2],oshape[3]);
            output_tensor->SetShape(GetShapeType(oshape));

            //quant output

            output_tensor->SetScale(docAct[(op_name_last + ":out").c_str()]["scale"][0].GetFloat());
            output_tensor->SetMax(docAct[(op_name_last + ":out").c_str()]["max_value"][0].GetFloat());
            output_tensor->SetMin(docAct[(op_name_last + ":out").c_str()]["min_value"][0].GetFloat());
			if  (std::strcmp(docAct[(op_name_last + ":out").c_str()]["quantizer"].GetString(),"Symmetric")==0){
				output_tensor->SetQuantType(kQT_Sym);
				output_tensor->SetZeroPoint(docAct[(op_name_last + ":out").c_str()]["zero_point"].GetFloat());}
		    else{
				output_tensor->SetQuantType(kQT_Asym);
				output_tensor->SetZeroPoint(docAct[(op_name_last + ":out").c_str()]["zero_point"][0].GetFloat());}

            decltype(inputs) outputs{op_name + ":out"};

            //quant weight

            std::string kernel_quant_name;
            kernel_quant_name = kernel_quant_name + "nn.conv2d" + "_weight_" + std::to_string(layer_idx) + "_0:in";
            kernel_tensor->SetName(kernel_quant_name);
            std::string bias_quant_name;
            bias_quant_name = bias_quant_name + "add" + "_bias_" + std::to_string(layer_idx+2) + "_0:in";
            bias_tensor->SetName(bias_quant_name);

            std::vector<float> channel_scale(kernel_shape.dim[0]);
            std::vector<int> channel_zero_point(kernel_shape.dim[0]);
            std::vector<float> channel_max(kernel_shape.dim[0]);
            std::vector<float> channel_min(kernel_shape.dim[0]);

            const rapidjson::Value& scale = docWgt[(kernel_quant_name).c_str()]["scale"];
            const rapidjson::Value& zero_point = docWgt[(kernel_quant_name).c_str()]["zero_point"];
            const rapidjson::Value& max = docWgt[(kernel_quant_name).c_str()]["max_value"];
            const rapidjson::Value& min = docWgt[(kernel_quant_name).c_str()]["min_value"];
            if (scale.Size()>1) {

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
                printf("set conv_exp perchannel!!!!\n");
                param.quant_mode = kQM_PerChannel;
                
				if  (std::strcmp(docWgt[(kernel_quant_name).c_str()]["quantizer"].GetString(),"Symmetric")==0)
					kernel_tensor->SetQuantType(kQT_Sym);
		    	else
					kernel_tensor->SetQuantType(kQT_Asym);
            }
			else {

                kernel_tensor->SetScale(docWgt[(kernel_quant_name).c_str()]["scale"][0].GetFloat());

				kernel_tensor->SetMax(docWgt[(kernel_quant_name).c_str()]["max_value"][0].GetFloat());
				kernel_tensor->SetMin(docWgt[(kernel_quant_name).c_str()]["min_value"][0].GetFloat());
                param.quant_mode = kQM_PerTensor;
				if  (std::strcmp(docWgt[(kernel_quant_name).c_str()]["quantizer"].GetString(),"Symmetric")==0){
					kernel_tensor->SetQuantType(kQT_Sym);
					kernel_tensor->SetZeroPoint(docWgt[(kernel_quant_name).c_str()]["zero_point"].GetFloat());}
		    	else{
					kernel_tensor->SetQuantType(kQT_Asym);
					kernel_tensor->SetZeroPoint(docWgt[(kernel_quant_name).c_str()]["zero_point"][0].GetFloat());}
            }

            //updata layer_idx for pattern
            layer_idx = layer_idx + 3;

            //CHECK_EQ(inputs.size(), func->num_inputs) << "wrong input arity: " << inputs.size();
            vis_tensor_map_[call] = outputs;

        }
        else if (pattern_name == "vis_vpu.dense_bias"){
            // build input tensors
            printf("Into vis_vpu.dense_bias Node.\n");
            std::vector<std::string> inputs;
            for (const auto& arg : cn->args) {
                CHECK(vis_tensor_map_.find(arg) != vis_tensor_map_.end());  // .at() gives bad message
                const auto& ts = vis_tensor_map_[arg];
                CHECK_EQ(ts.size(), 1) << "call arg is not expected be tuple";
                inputs.emplace_back(ts.back());
            }

            std::vector<std::string> op_namelist = {"add"};
            const CallNode* add_call = GetRootCall(func->body.as<CallNode>(), 0, op_namelist);//

            // set bias tensor data
            auto cnst = GetRef<Constant>(add_call->args[1].as<ConstantNode>());
            CHECK(vis_tensor_map_.find(cnst) == vis_tensor_map_.end());
            std::string name = "bias" + std::to_string(layer_idx + 1);  // temp name

            auto tt = add_call->args[1].as<ConstantNode>()->tensor_type();
            auto shape = GetConcrete<int>(tt->shape);

            auto data_container = add_call->args[1].as<ConstantNode>()->data.operator->();
            auto data_ptr = static_cast<uint8_t*>(data_container->data) + data_container->byte_offset;
            auto data_type = add_call->args[1].as<ConstantNode>()->data.DataType();

            TensorPtr bias_tensor = graph->CreateTensor(name, kTT_Const, Tvm2VisVpu(data_type));
            auto bias_shape = GetShapeType(shape);
            bias_tensor->SetShape(bias_shape);
            /// allocate memory
            int mem_size = bias_tensor->GetMemSize();
            bias_tensor->SetData(data_ptr, mem_size);

            //create node
            CHECK_EQ(inputs.size(), 2) << "number of inputs need to corrected";
            auto input_name = inputs.at(0);
            auto kernel_name = inputs.at(1);

            TensorPtr input_tensor = graph->FindTensor(input_name);
            TensorPtr kernel_tensor = graph->FindTensor(kernel_name);

            auto kernel_shape = kernel_tensor->GetShape();

            //fc or matmul
            auto input_shape = input_tensor->GetShape();
            int nozero = 0;
            for (int i=0; i<input_shape.dim_num;i++){
                if (input_shape.dim[i] > 1) {
                    nozero++;
                }
            }
            bool isfc = (nozero == 1);

            //create name
            std::string op_name = (isfc ? "fc_" : "matmul_") + std::to_string(layer_idx);
            std::string op_name_first = "nn.dense_" + std::to_string(layer_idx);
            std::string op_name_last = "add_" + std::to_string(layer_idx+1);

			if(bias_tensor->GetShape().Volume()==kernel_tensor->GetShape().dim[0])
			{
                vis::megrez::DataType out_type = Get_quant_type(op_name_last);
                TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

                NodePtr node = graph->CreateNode(op_name, isfc ? FC_OP : MATMUL_OP);//

                node->SetInputTensor(0, input_tensor);
                node->SetInputTensor(1, kernel_tensor);
                node->SetInputTensor(2, bias_tensor);
                node->SetOutputTensor(0, output_tensor);

                //set shape
                auto last_call = GetRef<Call>(add_call);
                auto oshape = tvm::relay::backend::GetShape(last_call->checked_type());
                output_tensor->SetShape(GetShapeType(oshape));

                //quant output
                output_tensor->SetScale(docAct[(op_name_last + ":out").c_str()]["scale"][0].GetFloat());

                output_tensor->SetMax(docAct[(op_name_last + ":out").c_str()]["max_value"][0].GetFloat());
                output_tensor->SetMin(docAct[(op_name_last + ":out").c_str()]["min_value"][0].GetFloat());
                if  (std::strcmp(docAct[(op_name_last + ":out").c_str()]["quantizer"].GetString(),"Symmetric")==0){
                    output_tensor->SetQuantType(kQT_Sym);
                    output_tensor->SetZeroPoint(docAct[(op_name_last + ":out").c_str()]["zero_point"].GetFloat());}
                else{
                    output_tensor->SetQuantType(kQT_Asym);
                    output_tensor->SetZeroPoint(docAct[(op_name_last + ":out").c_str()]["zero_point"][0].GetFloat());}

                decltype(inputs) outputs{op_name + ":out"};

                //quant weight
                if (kernel_tensor->GetTensorType() == kTT_Const) {
                    std::string kernel_quant_name;
                    kernel_quant_name =  kernel_quant_name + "nn.dense" + "_weight_" + std::to_string(layer_idx) + "_0:in";
                    kernel_tensor->SetName(kernel_quant_name);
                    std::string bias_quant_name;
                    bias_quant_name = bias_quant_name + "add" + "_bias_" + std::to_string(layer_idx+1) + "_0:in";
                    bias_tensor->SetName(bias_quant_name);

                    std::vector<float> channel_scale(kernel_shape.dim[0]);
                    std::vector<int> channel_zero_point(kernel_shape.dim[0]);
                    std::vector<float> channel_max(kernel_shape.dim[0]);
                    std::vector<float> channel_min(kernel_shape.dim[0]);

                    const rapidjson::Value& scale = docWgt[(kernel_quant_name).c_str()]["scale"];
                    const rapidjson::Value& zero_point = docWgt[(kernel_quant_name).c_str()]["zero_point"];
                    const rapidjson::Value& max = docWgt[(kernel_quant_name).c_str()]["max_value"];
                    const rapidjson::Value& min = docWgt[(kernel_quant_name).c_str()]["min_value"];
                    if (scale.Size()>1) {
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
                        if  (std::strcmp(docWgt[(kernel_quant_name).c_str()]["quantizer"].GetString(),"Symmetric")==0)
                            kernel_tensor->SetQuantType(kQT_Sym);
                        else
                            kernel_tensor->SetQuantType(kQT_Asym);
                    }
                    else {
                        kernel_tensor->SetScale(docWgt[(kernel_quant_name).c_str()]["scale"][0].GetFloat());

                        kernel_tensor->SetMax(docWgt[(kernel_quant_name).c_str()]["max_value"][0].GetFloat());
                        kernel_tensor->SetMin(docWgt[(kernel_quant_name).c_str()]["min_value"][0].GetFloat());
                        //param.quant_mode = kQM_PerTensor;
                        if  (std::strcmp(docWgt[(kernel_quant_name).c_str()]["quantizer"].GetString(),"Symmetric")==0){
                            kernel_tensor->SetQuantType(kQT_Sym);
                            kernel_tensor->SetZeroPoint(docWgt[(kernel_quant_name).c_str()]["zero_point"].GetFloat());}
                        else{
                            kernel_tensor->SetQuantType(kQT_Asym);
                            kernel_tensor->SetZeroPoint(docWgt[(kernel_quant_name).c_str()]["zero_point"][0].GetFloat());}
                    }
                }
//                 std::cout<<"support or not?????"<<std::endl;
//                 std::cout<<docTiling[(op_name_first).c_str()]["input_tile_pos"][0][1][1].GetInt()<<std::endl;
//                 bool support = (docTiling[(op_name_first).c_str()]["input_tile_pos"][0][1][1].GetInt() <= 8196);
//                  std::cout<<"support or not?????"<<std::endl;
//                 int find_group = docTiling[(op_name_first).c_str()]["group_index"].GetInt();
// std::cout<<"support or not?????"<<std::endl;
//                 auto find = tile_sort_map.find(find_group);
//                 if (find != tile_sort_map.end()) 
//                 {
//                     support = support && (!tile_sort_map[find_group]->dont_support);
//                 }
//                 std::cout<<"support or not?????"<<std::endl;

                if (doTiling)
                {
                    std::string tiling_name;
                    tiling_name = op_name_first;
                    int output_tile_num = docTiling[(tiling_name).c_str()]["output_tile_num"].GetInt();
                    int output_dim_num = docTiling[(tiling_name).c_str()]["output_dim_num"].GetInt();

                    int output_tile_dim = docTiling[(tiling_name).c_str()]["output_tile_dims"][0].GetInt();
                    std::vector<TileTensorShape> tile_shape_list;
                    std::vector<TileTensorShape> tile_shape_list_ri;
                    std::vector<TileTensorShape> tile_shape_list_ro;

                    int fc_len = docTiling[(tiling_name).c_str()]["input_tile_pos"][0][1][1].GetInt();
                    if (fc_len > 1600)
                    {
                        std::vector<int> tileShapeInput;
                        std::vector<int> tilePosInput;
                        tilePosInput.push_back(0);
                        tilePosInput.push_back(0);
                        tilePosInput.push_back(0);
                        tilePosInput.push_back(0);
                        tileShapeInput.push_back(1);
                        tileShapeInput.push_back(docTiling[(tiling_name).c_str()]["input_tile_pos"][0][1][1].GetInt());
                        tileShapeInput.push_back(1);
                        tileShapeInput.push_back(1);
                        TileTensorShape tile_tensor_shape_ri;
                        tile_tensor_shape_ri.shape = ShapeType(GetShapeType(tileShapeInput));
                        tile_tensor_shape_ri.pos = ShapeType(GetShapeType(tilePosInput));
                        tile_shape_list_ri.push_back(tile_tensor_shape_ri);
                        if (useTiling)
                            RegisterTileSchema(op_name + ":ri", tile_shape_list_ri);
                    }
                    
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
                            assert(oshape[i] == (docTiling[(tiling_name).c_str()]["output_tile_pos"][k][i][1].GetInt() - tilePos[i]));
                            tilePos.push_back(0);
                            tileShape.push_back(oshape[i]);
                            }
                        }
                        if (fc_len > 1600)
                        {
                            tilePos.push_back(0);
                            tilePos.push_back(0);
                            tileShape.push_back(1);
                            tileShape.push_back(1);
                        }
                        TileTensorShape tile_tensor_shape;
                        tile_tensor_shape.shape = ShapeType(GetShapeType(tileShape));
                        tile_tensor_shape.pos = ShapeType(GetShapeType(tilePos));
                        tile_shape_list.push_back(tile_tensor_shape);

                    }
                    if (useTiling)
                        RegisterTileSchema(op_name, tile_shape_list);
                    if (fc_len > 1600)
                    {
                        std::vector<int> tileShapeOutput;
                        std::vector<int> tilePosOutput;
                        tilePosOutput.push_back(0);
                        tilePosOutput.push_back(0);
                        tileShapeOutput.push_back(1);
                        tileShapeOutput.push_back(oshape[1]);
                        TileTensorShape tile_tensor_shape_ro;
                        tile_tensor_shape_ro.shape = ShapeType(GetShapeType(tileShapeOutput));
                        tile_tensor_shape_ro.pos = ShapeType(GetShapeType(tilePosOutput));
                        tile_shape_list_ro.push_back(tile_tensor_shape_ro);
                        if (useTiling)
                            RegisterTileSchema(op_name + ":ro", tile_shape_list_ro);
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
                        // tile_sort_map[group_index]->call_index_list.insert(tile_sort_map[group_index]->call_index_list.begin() + insert, call_index);
                        // tile_sort_map[group_index]->call_name_list.insert(tile_sort_map[group_index]->call_name_list.begin() + insert, op_name + ":ro");

                        tile_sort_map[group_index]->call_index_list.insert(tile_sort_map[group_index]->call_index_list.begin() + insert, call_index);
                        tile_sort_map[group_index]->call_name_list.insert(tile_sort_map[group_index]->call_name_list.begin() + insert, op_name);
                        if (fc_len > 1600)
                        {
                            tile_sort_map[group_index]->pre_process.push_back(op_name + ":ri");
                            tile_sort_map[group_index]->post_process.push_back(op_name + ":ro");
                        }
                        // tile_sort_map[group_index]->call_index_list.insert(tile_sort_map[group_index]->call_index_list.begin() + insert, call_index);
                        // tile_sort_map[group_index]->call_name_list.insert(tile_sort_map[group_index]->call_name_list.begin() + insert, op_name + ":ri");
                    }
                    else
                    {
                        GroupInfoPtr group_info = GroupInfoPtr(new GroupInfo);
                        group_info->index = group_index;
                        group_info->tile_num = output_tile_num;

                        // group_info->call_index_list.push_back(call_index);
                        // group_info->call_name_list.push_back(op_name + ":ri");

                        group_info->call_index_list.push_back(call_index);
                        group_info->call_name_list.push_back(op_name);
                        if (fc_len > 1600)
                        {
                            group_info->pre_process.push_back(op_name + ":ri");
                            group_info->post_process.push_back(op_name + ":ro");
                        }
                        // group_info->call_index_list.push_back(call_index);
                        // group_info->call_name_list.push_back(op_name + ":ro");

                        tile_sort_map[group_index] = group_info;
                    }
                }

                if (doAllocate)
                {
                    printf("***********dense doAllocate************\n");
                    std::string op_input_name = input_tensor->GetName();
                    std::string op_output_name = output_tensor->GetName();
                    int fc_len = docTiling[(op_name_first).c_str()]["input_tile_pos"][0][1][1].GetInt();
                    if (fc_len > 1600)
                    {
                        std::string tile_name = op_name + ":ri.tile0";
                        mem_allocator_builder_->CreateTile(tile_name, 0);
                        std::string pre_input_name = op_input_name;
                        std::string pre_output_name = op_input_name + ":parent:" +  op_name + ":ri";
                        op_input_name = pre_output_name;
                        std::vector<int> inputTileShape;
                        std::vector<int> inputTilePos;
                        inputTilePos.push_back(0);
                        inputTilePos.push_back(0);
                        inputTileShape.push_back(1);
                        inputTileShape.push_back(docTiling[(op_name_first).c_str()]["input_tile_pos"][0][1][1].GetInt());

                        std::string tile_input_name = pre_input_name;
                        for (int i = 0; i < 2; i++)
                        {
                            if (i==0)
                                tile_input_name += "@";
                            else
                                tile_input_name += ",";
                            tile_input_name = tile_input_name + std::to_string(inputTilePos[i]);
                        }
                        for (int i = 0; i < 2; i++)
                        {
                            if (i==0)
                                tile_input_name += ":";
                            else
                                tile_input_name += ",";
                            tile_input_name = tile_input_name + std::to_string(inputTileShape[i]);
                        }
                        int input_bit_width = docAct[(input_tensor->GetName()).c_str()]["bit_width"].GetInt();
                        int input_block_type = BLOCK_TYPE_INPUT;
                        mem_allocator_builder_->CreateBlock(tile_name, pre_input_name, 
                            tile_input_name, inputTileShape, inputTilePos,
                            input_bit_width/8, input_block_type);

                        // create output tensor
                        printf("***********create pre output tensor************\n");
                        std::string tile_output_name = pre_output_name;
                        std::vector<int> outputTileShape;
                        std::vector<int> outputTilePos;
                        outputTilePos.push_back(0);
                        outputTilePos.push_back(0);
                        outputTilePos.push_back(0);
                        outputTilePos.push_back(0);
                        outputTileShape.push_back(1);
                        outputTileShape.push_back(docTiling[(op_name_first).c_str()]["input_tile_pos"][0][1][1].GetInt());
                        outputTileShape.push_back(1);
                        outputTileShape.push_back(1);
                        for (int i = 0; i < 4; i++)
                        {
                            if (i==0)
                                tile_output_name += "@";
                            else
                                tile_output_name += ",";
                            tile_output_name = tile_output_name + std::to_string(outputTilePos[i]);
                        }
                        for (int i = 0; i < 4; i++)
                        {
                            if (i==0)
                                tile_output_name += ":";
                            else
                                tile_output_name += ",";
                            tile_output_name = tile_output_name + std::to_string(outputTileShape[i]);
                        }
                        int output_bit_width = input_bit_width;
                        int output_block_type = BLOCK_TYPE_OUTPUT;
                        mem_allocator_builder_->CreateBlock(tile_name, pre_output_name, 
                            tile_output_name, outputTileShape, outputTilePos,
                            output_bit_width/8, output_block_type);

                        mem_allocator_builder_->CreateReuse(tile_name, tile_input_name, tile_output_name, true);
                    }

                    if (fc_len > 1600)
                    {
                        std::string tile_name = op_name + ":ro.tile0";
                        mem_allocator_builder_->CreateTile(tile_name, 0);
                        std::string post_input_name = op_output_name + ":child:" + op_name + ":ro";
                        std::string post_output_name = op_output_name;
                        op_output_name = post_input_name;
                        printf("***********create pre input tensor************\n");
                        std::vector<int> inputTileShape;
                        std::vector<int> inputTilePos;
                        inputTilePos.push_back(0);
                        inputTilePos.push_back(0);
                        inputTilePos.push_back(0);
                        inputTilePos.push_back(0);
                        inputTileShape.push_back(1);
                        inputTileShape.push_back(oshape[1]);
                        inputTileShape.push_back(1);
                        inputTileShape.push_back(1);
                        std::string tile_input_name = post_input_name;
                        for (int i = 0; i < 4; i++)
                        {
                            if (i==0)
                                tile_input_name += "@";
                            else
                                tile_input_name += ",";
                            tile_input_name = tile_input_name + std::to_string(inputTilePos[i]);
                        }
                        for (int i = 0; i < 4; i++)
                        {
                            if (i==0)
                                tile_input_name += ":";
                            else
                                tile_input_name += ",";
                            tile_input_name = tile_input_name + std::to_string(inputTileShape[i]);
                        }
                        int input_bit_width = docAct[(input_tensor->GetName()).c_str()]["bit_width"].GetInt();
                        int input_block_type = BLOCK_TYPE_INPUT;
                        mem_allocator_builder_->CreateBlock(tile_name, post_input_name, 
                            tile_input_name, inputTileShape, inputTilePos,
                            input_bit_width/8, input_block_type);

                        // create output tensor
                        printf("***********create pre output tensor************\n");
                        std::string tile_output_name = post_output_name;
                        std::vector<int> outputTileShape;
                        std::vector<int> outputTilePos;
                        outputTilePos.push_back(0);
                        outputTilePos.push_back(0);
                        outputTileShape.push_back(1);
                        outputTileShape.push_back(oshape[1]);
                        for (int i = 0; i < 2; i++)
                        {
                            if (i==0)
                                tile_output_name += "@";
                            else
                                tile_output_name += ",";
                            tile_output_name = tile_output_name + std::to_string(outputTilePos[i]);
                        }
                        for (int i = 0; i < 2; i++)
                        {
                            if (i==0)
                                tile_output_name += ":";
                            else
                                tile_output_name += ",";
                            tile_output_name = tile_output_name + std::to_string(outputTileShape[i]);
                        }
                        int output_bit_width = input_bit_width;
                        int output_block_type = BLOCK_TYPE_OUTPUT;
                        if (layer_idx + 2 == docTiling["network"]["call_num"].GetInt())
                        {
                            output_block_type = BLOCK_TYPE_NET_OUTPUT;
                        }
                        mem_allocator_builder_->CreateBlock(tile_name, post_output_name, 
                            tile_output_name, outputTileShape, outputTilePos,
                            output_bit_width/8, output_block_type);

                        mem_allocator_builder_->CreateReuse(tile_name, tile_input_name, tile_output_name, true);
                    }

                    int output_tile_num = docTiling[(op_name_first).c_str()]["output_tile_num"].GetInt();
                    int output_dim_num = docTiling[(op_name_first).c_str()]["output_dim_num"].GetInt();

                    // int output_tile_dim = docTiling[(op_name_first).c_str()]["output_tile_dims"][0].GetInt();
                    
                    if (fc_len > 1600)
                    {
                        for (int k = 0; k < output_tile_num; k++)
                        {
                            std::string op_kernel_name = kernel_tensor->GetName();
                            std::string op_bias_name = bias_tensor->GetName();
                            std::string tile_name = op_name + ".tile" + std::to_string(k);
                            mem_allocator_builder_->CreateTile(tile_name, 0);

                            // create input tensor
                            printf("***********create input tensor************\n");
                            std::vector<int> inputTileShape;
                            std::vector<int> inputTilePos;
                            for (int i = 0; i < output_dim_num; i++)
                            {
                                int index = docTiling[(op_name_first).c_str()]["input_tile_pos"].Size() == 1 ? 0 : k;
                                inputTilePos.push_back(docTiling[(op_name_first).c_str()]["input_tile_pos"][index][i][0].GetInt());
                                inputTileShape.push_back(docTiling[(op_name_first).c_str()]["input_tile_pos"][index][i][1].GetInt() - inputTilePos[i]);
                            }
                            inputTilePos.push_back(0);
                            inputTilePos.push_back(0);
                            inputTileShape.push_back(1);
                            inputTileShape.push_back(1);
                            std::string tile_input_name = op_input_name;
                            for (int i = 0; i < 4; i++)
                            {
                                if (i==0)
                                    tile_input_name += "@";
                                else
                                    tile_input_name += ",";
                                tile_input_name = tile_input_name + std::to_string(inputTilePos[i]);
                            }
                            for (int i = 0; i < 4; i++)
                            {
                                if (i==0)
                                    tile_input_name += ":";
                                else
                                    tile_input_name += ",";
                                tile_input_name = tile_input_name + std::to_string(inputTileShape[i]);
                            }
                            int input_bit_width = docAct[(input_tensor->GetName()).c_str()]["bit_width"].GetInt();
                            int input_block_type = BLOCK_TYPE_INPUT;
                            mem_allocator_builder_->CreateBlock(tile_name, op_input_name, 
                                tile_input_name, inputTileShape, inputTilePos,
                                input_bit_width/8, input_block_type);

                            // create kernel tensor
                            printf("***********create kernel tensor************\n");
                            std::vector<int> kernelTileShape;
                            std::vector<int> kernelTilePos;
                            for (int i = 0; i < 2; i++)
                            {
                                int index = docTiling[(op_name_first).c_str()]["weight_tile_pos"].Size() == 1 ? 0 : k;

                                
                                kernelTileShape.push_back(docTiling[(op_name_first).c_str()]["weight_tile_pos"][index][i][1].GetInt() - docTiling[(op_name_first).c_str()]["weight_tile_pos"][index][i][0].GetInt());
                                
                                if (i == 0)
                                {
                                    kernelTileShape[i] = ((kernelTileShape[i] % 32) == 0) ? kernelTileShape[i] : (kernelTileShape[i] / 32 + 1) * 32;
                                    op_kernel_name = op_kernel_name + ":" + std::to_string(docTiling[(op_name_first).c_str()]["weight_tile_pos"][index][i][0].GetInt());
                                    op_kernel_name = op_kernel_name + ":" + std::to_string(docTiling[(op_name_first).c_str()]["weight_tile_pos"][index][i][1].GetInt());
                                }
                            }
                            kernelTilePos.push_back(0);
                            kernelTilePos.push_back(0);
                            kernelTilePos.push_back(0);
                            kernelTilePos.push_back(0);
                            kernelTileShape.push_back(1);
                            kernelTileShape.push_back(1);
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
                            int index = docTiling[(op_name_last).c_str()]["weight_tile_pos"].Size() == 1 ? 0 : k;
                            biasTilePos.push_back(0);
                            biasTileShape.push_back(docTiling[(op_name_last).c_str()]["weight_tile_pos"][index][0][1].GetInt() - docTiling[(op_name_last).c_str()]["weight_tile_pos"][index][0][0].GetInt());
                            biasTileShape[0] = ((biasTileShape[0] % 64) == 0) ? (biasTileShape[0] / 64) * 352 : (biasTileShape[0] / 64 + 1) * 352;
                            op_bias_name = op_bias_name + ":" + std::to_string(docTiling[(op_name_last).c_str()]["weight_tile_pos"][index][0][0].GetInt());
                            op_bias_name = op_bias_name + ":" + std::to_string(docTiling[(op_name_last).c_str()]["weight_tile_pos"][index][0][1].GetInt());
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
                                outputTilePos.push_back(docTiling[(op_name_first).c_str()]["output_tile_pos"][k][i][0].GetInt());
                                outputTileShape.push_back(docTiling[(op_name_first).c_str()]["output_tile_pos"][k][i][1].GetInt() - outputTilePos[i]);
                            }
                            outputTilePos.push_back(0);
                            outputTilePos.push_back(0);
                            outputTileShape.push_back(1);
                            outputTileShape.push_back(1);
                            for (int i = 0; i < 4; i++)
                            {
                                if (i==0)
                                    tile_output_name += "@";
                                else
                                    tile_output_name += ",";
                                tile_output_name = tile_output_name + std::to_string(outputTilePos[i]);
                            }
                            for (int i = 0; i < 4; i++)
                            {
                                if (i==0)
                                    tile_output_name += ":";
                                else
                                    tile_output_name += ",";
                                tile_output_name = tile_output_name + std::to_string(outputTileShape[i]);
                            }
                            int output_bit_width = docAct[(input_tensor->GetName()).c_str()]["bit_width"].GetInt();
                            int output_block_type = BLOCK_TYPE_OUTPUT;
                            if (layer_idx + 2 == docTiling["network"]["call_num"].GetInt() and fc_len<=1600)
                            {
                                output_block_type = BLOCK_TYPE_NET_OUTPUT;
                            }
                            mem_allocator_builder_->CreateBlock(tile_name, op_output_name, 
                                tile_output_name, outputTileShape, outputTilePos,
                                output_bit_width/8, output_block_type);

                        }
                    }
                    else
                    {
                        for (int k = 0; k < output_tile_num; k++)
                        {
                            std::string op_kernel_name = kernel_tensor->GetName();
                            std::string op_bias_name = bias_tensor->GetName();
                            std::string tile_name = op_name + ".tile" + std::to_string(k);
                            mem_allocator_builder_->CreateTile(tile_name, 0);

                            // create input tensor
                            printf("***********create input tensor************\n");
                            std::vector<int> inputTileShape;
                            std::vector<int> inputTilePos;
                            for (int i = 0; i < output_dim_num; i++)
                            {
                                int index = docTiling[(op_name_first).c_str()]["input_tile_pos"].Size() == 1 ? 0 : k;
                                inputTilePos.push_back(docTiling[(op_name_first).c_str()]["input_tile_pos"][index][i][0].GetInt());
                                inputTileShape.push_back(docTiling[(op_name_first).c_str()]["input_tile_pos"][index][i][1].GetInt() - inputTilePos[i]);
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
                            int input_bit_width = docAct[(input_tensor->GetName()).c_str()]["bit_width"].GetInt();
                            int input_block_type = BLOCK_TYPE_INPUT;
                            mem_allocator_builder_->CreateBlock(tile_name, op_input_name, 
                                tile_input_name, inputTileShape, inputTilePos,
                                input_bit_width/8, input_block_type);

                            // create kernel tensor
                            printf("***********create kernel tensor************\n");
                            std::vector<int> kernelTileShape;
                            std::vector<int> kernelTilePos;
                            for (int i = 0; i < output_dim_num; i++)
                            {
                                int index = docTiling[(op_name_first).c_str()]["weight_tile_pos"].Size() == 1 ? 0 : k;

                                
                                kernelTileShape.push_back(docTiling[(op_name_first).c_str()]["weight_tile_pos"][index][i][1].GetInt() - docTiling[(op_name_first).c_str()]["weight_tile_pos"][index][i][0].GetInt());
                                
                                if (i == 0)
                                {
                                    kernelTileShape[i] = ((kernelTileShape[i] % 64) == 0) ? kernelTileShape[i] : (kernelTileShape[i] / 64 + 1) * 64;
                                    op_kernel_name = op_kernel_name + ":" + std::to_string(docTiling[(op_name_first).c_str()]["weight_tile_pos"][index][i][0].GetInt());
                                    op_kernel_name = op_kernel_name + ":" + std::to_string(docTiling[(op_name_first).c_str()]["weight_tile_pos"][index][i][1].GetInt() - docTiling[(op_name_first).c_str()]["weight_tile_pos"][index][i][0].GetInt());
                                }
                            }
                            kernelTilePos.push_back(0);
                            kernelTilePos.push_back(0);

                            std::string tile_kernel_name = op_kernel_name;
                            for (int i = 0; i < output_dim_num; i++)
                            {
                                if (i==0)
                                    tile_kernel_name += "@";
                                else
                                    tile_kernel_name += ",";
                                tile_kernel_name = tile_kernel_name + std::to_string(kernelTilePos[i]);
                            }
                            for (int i = 0; i < output_dim_num; i++)
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
                            int index = docTiling[(op_name_last).c_str()]["weight_tile_pos"].Size() == 1 ? 0 : k;
                            biasTilePos.push_back(0);
                            biasTileShape.push_back(docTiling[(op_name_last).c_str()]["weight_tile_pos"][index][0][1].GetInt() - docTiling[(op_name_last).c_str()]["weight_tile_pos"][index][0][0].GetInt());
                            biasTileShape[0] = ((biasTileShape[0] % 64) == 0) ? (biasTileShape[0] / 64) * 352 : (biasTileShape[0] / 64 + 1) * 352;
                            op_bias_name = op_bias_name + ":" + std::to_string(docTiling[(op_name_last).c_str()]["weight_tile_pos"][index][0][0].GetInt());
                            op_bias_name = op_bias_name + ":" + std::to_string(docTiling[(op_name_last).c_str()]["weight_tile_pos"][index][0][1].GetInt() - docTiling[(op_name_last).c_str()]["weight_tile_pos"][index][0][0].GetInt());
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
                                outputTilePos.push_back(docTiling[(op_name_first).c_str()]["output_tile_pos"][k][i][0].GetInt());
                                outputTileShape.push_back(docTiling[(op_name_first).c_str()]["output_tile_pos"][k][i][1].GetInt() - outputTilePos[i]);
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
                            int output_bit_width = docAct[(input_tensor->GetName()).c_str()]["bit_width"].GetInt();
                            int output_block_type = BLOCK_TYPE_OUTPUT;
                            if (layer_idx + 2 == docTiling["network"]["call_num"].GetInt())
                            {
                                output_block_type = BLOCK_TYPE_NET_OUTPUT;
                            }
                            mem_allocator_builder_->CreateBlock(tile_name, op_output_name, 
                                tile_output_name, outputTileShape, outputTilePos,
                                output_bit_width/8, output_block_type);

                        }
                    }
                }
                
                // else
                // {
                //     printf("dont support!!!!!!!!!!!!!!!!!!!!!\n");
                //     int group_index = docTiling[(op_name_first).c_str()]["group_index"].GetInt();
                    
                //     GroupInfoPtr group_info = GroupInfoPtr(new GroupInfo);
                //     group_info->index = group_index;
                //     group_info->dont_support = true;
                //     // group_info->call_index_list.push_back(call_index);
                //     // group_info->call_name_list.push_back(op_name + ":ro");

                //     tile_sort_map[group_index] = group_info;
                // }

                layer_idx = layer_idx + 2;
                vis_tensor_map_[call] = outputs;

            }
			else
			{
                std::cout<<"not bias add dense "<<op_name_first<<std::endl;
                assert(0);
                vis::megrez::DataType out_type = Get_quant_type(op_name_first);
                TensorPtr dense_output_tensor = graph->CreateTensor(op_name + ":_dence_out", kTT_Var, out_type);

                NodePtr node = graph->CreateNode(op_name, isfc ? FC_OP : MATMUL_OP);//

                node->SetInputTensor(0, input_tensor);
                node->SetInputTensor(1, kernel_tensor);
                node->SetOutputTensor(0, dense_output_tensor);

                //set shape
                std::vector<std::string> op_namelist = {"nn.dense","add"};
                const CallNode* dense_call = GetRootCall(func->body.as<CallNode>(), 1, op_namelist);//

                auto oshape = tvm::relay::backend::GetShape(dense_call->checked_type());

                dense_output_tensor->SetShape(GetShapeType(oshape));

                //quant output
                dense_output_tensor->SetScale(docAct[(op_name_first + ":out").c_str()]["scale"][0].GetFloat());

                dense_output_tensor->SetMax(docAct[(op_name_first + ":out").c_str()]["max_value"][0].GetFloat());
                dense_output_tensor->SetMin(docAct[(op_name_first + ":out").c_str()]["min_value"][0].GetFloat());
                if  (std::strcmp(docAct[(op_name_first + ":out").c_str()]["quantizer"].GetString(),"Symmetric")==0){
                    dense_output_tensor->SetQuantType(kQT_Sym);
                    dense_output_tensor->SetZeroPoint(docAct[(op_name_first + ":out").c_str()]["zero_point"].GetFloat());}
                else{
                    dense_output_tensor->SetQuantType(kQT_Asym);
                    dense_output_tensor->SetZeroPoint(docAct[(op_name_first + ":out").c_str()]["zero_point"][0].GetFloat());}


                //quant weight
                if (kernel_tensor->GetTensorType() == kTT_Const) {
                    std::string kernel_quant_name;
                    kernel_quant_name =  kernel_quant_name + "nn.dense" + "_weight_" + std::to_string(layer_idx) + "_0:in";
                    kernel_tensor->SetName(kernel_quant_name);

                    std::vector<float> channel_scale(kernel_shape.dim[0]);
                    std::vector<int> channel_zero_point(kernel_shape.dim[0]);
                    std::vector<float> channel_max(kernel_shape.dim[0]);
                    std::vector<float> channel_min(kernel_shape.dim[0]);

                    const rapidjson::Value& scale = docWgt[(kernel_quant_name).c_str()]["scale"];
                    const rapidjson::Value& zero_point = docWgt[(kernel_quant_name).c_str()]["zero_point"];
                    const rapidjson::Value& max = docWgt[(kernel_quant_name).c_str()]["max_value"];
                    const rapidjson::Value& min = docWgt[(kernel_quant_name).c_str()]["min_value"];

                    if (scale.Size()>1) {
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
                        if  (std::strcmp(docWgt[(kernel_quant_name).c_str()]["quantizer"].GetString(),"Symmetric")==0)
                            kernel_tensor->SetQuantType(kQT_Sym);
                        else
                            kernel_tensor->SetQuantType(kQT_Asym);
                    }
                    else {
                        kernel_tensor->SetScale(docWgt[(kernel_quant_name).c_str()]["scale"][0].GetFloat());

                        kernel_tensor->SetMax(docWgt[(kernel_quant_name).c_str()]["max_value"][0].GetFloat());
                        kernel_tensor->SetMin(docWgt[(kernel_quant_name).c_str()]["min_value"][0].GetFloat());
                        //param.quant_mode = kQM_PerTensor;
                        if  (std::strcmp(docWgt[(kernel_quant_name).c_str()]["quantizer"].GetString(),"Symmetric")==0){
                            kernel_tensor->SetQuantType(kQT_Sym);
                            kernel_tensor->SetZeroPoint(docWgt[(kernel_quant_name).c_str()]["zero_point"].GetFloat());}
                        else{
                            kernel_tensor->SetQuantType(kQT_Asym);
                            kernel_tensor->SetZeroPoint(docWgt[(kernel_quant_name).c_str()]["zero_point"][0].GetFloat());}
                    }
                }

                NodePtr add_node = graph->CreateNode(op_name_last, ADD_OP);//

                vis::megrez::DataType add_out_type = Get_quant_type(op_name_last);
                TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, add_out_type);

                add_node->SetInputTensor(0, dense_output_tensor);
                add_node->SetInputTensor(1, bias_tensor);
                add_node->SetOutputTensor(0, output_tensor);

                std::vector<std::string> op_namelist_1 = {"add"};
                const CallNode* add_call = GetRootCall(func->body.as<CallNode>(), 0, op_namelist_1);
                auto last_call = GetRef<Call>(add_call);

                auto add_oshape = tvm::relay::backend::GetShape(last_call->checked_type());

                output_tensor->SetShape(GetShapeType(add_oshape));

                //quant output
                output_tensor->SetScale(docAct[(op_name_last + ":out").c_str()]["scale"][0].GetFloat());

                output_tensor->SetMax(docAct[(op_name_last + ":out").c_str()]["max_value"][0].GetFloat());
                output_tensor->SetMin(docAct[(op_name_last + ":out").c_str()]["min_value"][0].GetFloat());
                if	(std::strcmp(docAct[(op_name_last + ":out").c_str()]["quantizer"].GetString(),"Symmetric")==0){
                    output_tensor->SetQuantType(kQT_Sym);
                    output_tensor->SetZeroPoint(docAct[(op_name_last + ":out").c_str()]["zero_point"].GetFloat());}
                else{
                    output_tensor->SetQuantType(kQT_Asym);
                    output_tensor->SetZeroPoint(docAct[(op_name_last + ":out").c_str()]["zero_point"][0].GetFloat());}

                decltype(inputs) outputs{op_name + ":out"};

                layer_idx = layer_idx + 2;
                vis_tensor_map_[call] = outputs;

		   }

            //CHECK_EQ(inputs.size(), func->num_inputs) << "wrong input arity: " << inputs.size();
                        //updata layer_idx for pattern

        }
        else if (pattern_name == "vis_vpu.silu"){
            printf("Into vis_vpu.silu Node.\n");
            // build input tensors
            std::vector<std::string> inputs;
            for (const auto& arg : cn->args) {
                CHECK(vis_tensor_map_.find(arg) != vis_tensor_map_.end());  // .at() gives bad message
                const auto& ts = vis_tensor_map_[arg];
                CHECK_EQ(ts.size(), 1) << "call arg is not expected be tuple";
                inputs.emplace_back(ts.back());
            }
            //build name
            std::string op_name = "silu_" + std::to_string(layer_idx);
            std::string op_name_first = "sigmoid_" + std::to_string(layer_idx);
            std::string op_name_last = "multiply_" + std::to_string(layer_idx+1);

            //create node
            // CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";//
            auto input_name = inputs.at(0);

            TensorPtr input_tensor = graph->FindTensor(input_name);

            vis::megrez::DataType out_type = Get_quant_type(op_name_last);
            TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

            NodePtr node = graph->CreateNode(op_name, SILU_OP);//

            node->SetInputTensor(0, input_tensor);
            node->SetOutputTensor(0, output_tensor);

            //set shape
            std::vector<std::string> op_namelist = {"multiply"};
            const CallNode* mul_call = GetRootCall(func->body.as<CallNode>(), 0, op_namelist);
            auto last_call = GetRef<Call>(mul_call);
            auto oshape = tvm::relay::backend::GetShape(last_call->checked_type());
            output_tensor->SetShape(GetShapeType(oshape));

            //quant output
            output_tensor->SetScale(docAct[(op_name_last + ":out").c_str()]["scale"][0].GetFloat());

            output_tensor->SetMax(docAct[(op_name_last + ":out").c_str()]["max_value"][0].GetFloat());
            output_tensor->SetMin(docAct[(op_name_last + ":out").c_str()]["min_value"][0].GetFloat());
			if  (std::strcmp(docAct[(op_name_last + ":out").c_str()]["quantizer"].GetString(),"Symmetric")==0){
				output_tensor->SetQuantType(kQT_Sym);
				output_tensor->SetZeroPoint(docAct[(op_name_last + ":out").c_str()]["zero_point"].GetFloat());}
		    else{
				output_tensor->SetQuantType(kQT_Asym);
				output_tensor->SetZeroPoint(docAct[(op_name_last + ":out").c_str()]["zero_point"][0].GetFloat());}

            decltype(inputs) outputs{op_name + ":out"};

            if (doTiling)
            {
                std::string tiling_name;
                tiling_name = op_name_last;
                int output_tile_num = docTiling[(tiling_name).c_str()]["output_tile_num"].GetInt();
                int output_dim_num = docTiling[(tiling_name).c_str()]["output_dim_num"].GetInt();
                assert(output_dim_num == 4);
                int output_tile_dim = docTiling[(tiling_name).c_str()]["output_tile_dims"][0].GetInt();
                std::vector<TileTensorShape> tile_shape_list;
                printf("***********silu************\n");
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
            //updata layer_idx for pattern
            layer_idx = layer_idx + 2;

            //CHECK_EQ(inputs.size(), func->num_inputs) << "wrong input arity: " << inputs.size();
            vis_tensor_map_[call] = outputs;
        }
        else if (pattern_name == "vis_vpu.silu_simple"){
            printf("Into vis_vpu.silu Node.\n");
            // build input tensors
            std::vector<std::string> inputs;
            for (const auto& arg : cn->args) {
                CHECK(vis_tensor_map_.find(arg) != vis_tensor_map_.end());  // .at() gives bad message
                const auto& ts = vis_tensor_map_[arg];
                CHECK_EQ(ts.size(), 1) << "call arg is not expected be tuple";
                inputs.emplace_back(ts.back());
            }
            //build name
            std::string op_name = "silu_" + std::to_string(layer_idx);
            std::string op_name_first = "sigmoid_" + std::to_string(layer_idx);
            std::string op_name_last = "multiply_" + std::to_string(layer_idx+1);

            //create node
            // CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";//
            auto input_name = inputs.at(0);

            TensorPtr input_tensor = graph->FindTensor(input_name);

            vis::megrez::DataType out_type = Get_quant_type(op_name_last);
            TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

            NodePtr node = graph->CreateNode(op_name, SILU_OP);//

            node->SetInputTensor(0, input_tensor);
            node->SetOutputTensor(0, output_tensor);

            //set shape
            std::vector<std::string> op_namelist = {"multiply"};
            const CallNode* mul_call = GetRootCall(func->body.as<CallNode>(), 0, op_namelist);
            auto last_call = GetRef<Call>(mul_call);
            auto oshape = tvm::relay::backend::GetShape(last_call->checked_type());
            output_tensor->SetShape(GetShapeType(oshape));

            //quant output
            output_tensor->SetScale(docAct[(op_name_last + ":out").c_str()]["scale"][0].GetFloat());

            output_tensor->SetMax(docAct[(op_name_last + ":out").c_str()]["max_value"][0].GetFloat());
            output_tensor->SetMin(docAct[(op_name_last + ":out").c_str()]["min_value"][0].GetFloat());
			if  (std::strcmp(docAct[(op_name_last + ":out").c_str()]["quantizer"].GetString(),"Symmetric")==0){
				output_tensor->SetQuantType(kQT_Sym);
				output_tensor->SetZeroPoint(docAct[(op_name_last + ":out").c_str()]["zero_point"].GetFloat());}
		    else{
				output_tensor->SetQuantType(kQT_Asym);
				output_tensor->SetZeroPoint(docAct[(op_name_last + ":out").c_str()]["zero_point"][0].GetFloat());}

            decltype(inputs) outputs{op_name + ":out"};
            if (doTiling)
            {
                std::string tiling_name;
                tiling_name = op_name_last;
                int output_tile_num = docTiling[(tiling_name).c_str()]["output_tile_num"].GetInt();
                int output_dim_num = docTiling[(tiling_name).c_str()]["output_dim_num"].GetInt();
                assert(output_dim_num == 4);
                int output_tile_dim = docTiling[(tiling_name).c_str()]["output_tile_dims"][0].GetInt();
                std::vector<TileTensorShape> tile_shape_list;
                printf("***********silu_simple************\n");
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
            //updata layer_idx for pattern
            layer_idx = layer_idx + 2;

            //CHECK_EQ(inputs.size(), func->num_inputs) << "wrong input arity: " << inputs.size();
            vis_tensor_map_[call] = outputs;
        }
        else if(pattern_name == "vis_vpu.convTranspose_bias"){
            // build input tensors
            printf("Into vis_vpu.convTranspose_bias.\n");
            std::vector<std::string> inputs;
            printf("cn->args: %ld", cn->args.size());
            for (const auto& arg : cn->args) {
                CHECK(vis_tensor_map_.find(arg) != vis_tensor_map_.end());  // .at() gives bad message
                const auto& ts = vis_tensor_map_[arg];
                CHECK_EQ(ts.size(), 1) << "call arg is not expected be tuple";
                inputs.emplace_back(ts.back());
            }

            std::vector<std::string> op_namelist = {"nn.conv2d_transpose", "add"};
            const CallNode* conv_call = GetRootCall(func->body.as<CallNode>(), 1, op_namelist);
            std::vector<std::string> op_namelist_1 = {"add"};
            const CallNode* add_call = GetRootCall(func->body.as<CallNode>(), 0, op_namelist_1);

            // set bias tensor data
            auto cnst = GetRef<Constant>(add_call->args[1].as<ConstantNode>());
            CHECK(vis_tensor_map_.find(cnst) == vis_tensor_map_.end());
            std::string name = "bias" + std::to_string(layer_idx + 1);  // temp name

            auto tt = add_call->args[1].as<ConstantNode>()->tensor_type();
            auto shape = GetConcrete<int>(tt->shape);

            auto data_container = add_call->args[1].as<ConstantNode>()->data.operator->();
            auto data_ptr = static_cast<uint8_t*>(data_container->data) + data_container->byte_offset;
            auto data_type = add_call->args[1].as<ConstantNode>()->data.DataType();

            TensorPtr bias_tensor = graph->CreateTensor(name, kTT_Const, Tvm2VisVpu(data_type));
            auto bias_shape = GetShapeType(shape);
            bias_tensor->SetShape(bias_shape);
            /// allocate memory
            int mem_size = bias_tensor->GetMemSize();
            bias_tensor->SetData(data_ptr, mem_size);

            //create name

            std::string op_name = "nn.conv2d_transpose_" + std::to_string(layer_idx);
            std::string op_name_last = "add_" + std::to_string(layer_idx+1);

            //create node

            CHECK_EQ(inputs.size(), 2) << "number of inputs need to corrected";
            auto input_name = inputs.at(0);
            auto kernel_name = inputs.at(1);

            TensorPtr input_tensor = graph->FindTensor(input_name);
            TensorPtr kernel_tensor = graph->FindTensor(kernel_name);

            vis::megrez::DataType out_type = Get_quant_type(op_name_last);

            TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

            NodePtr node = graph->CreateNode(op_name, CONV_TRANSPOSE_OP);//

            node->SetInputTensor(0, input_tensor);
            node->SetInputTensor(1, kernel_tensor);
            node->SetInputTensor(2, bias_tensor);
            node->SetOutputTensor(0, output_tensor);

            auto kernel_shape = kernel_tensor->GetShape();

            //parse attr
            auto first_call = GetRef<Call>(conv_call);
            const auto* attrs = first_call->attrs.as<Conv2DTransposeAttrs>();//
            // std::cout<<"attrs=="<<attrs<<std::endl;

            CHECK(attrs) << "not ConvTranspose2DAttrs node";//

            CHECK_EQ(attrs->data_layout, "NCHW") << "layout " << attrs->data_layout << " not supported";//

            auto strides = GetConcrete<int>(attrs->strides);
            auto padding = GetConcrete<int>(attrs->padding);
            // auto dilations = GetConcrete<int>(attrs->dilation);
            auto kernel_size = GetConcrete<int>(attrs->kernel_size);

            int groups = attrs->groups;

            // config param

            ConvTransposeOpPtr op_ptr = std::dynamic_pointer_cast<ConvTransposeOp>(node->GetOp());//

            ConvTransposeParam& param = op_ptr->GetParam();//

            param.pad_h0 = padding[0];
            param.pad_h1 = padding[1];
            param.pad_w0 = padding[2];
            param.pad_w1 = padding[3];
            param.kernel_h = kernel_size[0];
            param.kernel_w = kernel_size[1];
            param.stride_h = strides[0];
            param.stride_w = strides[1];
            // param.dilation_h = dilations[0];
            // param.dilation_w = dilations[1];
            param.input_channel = kernel_shape.dim[1];
            param.output_channel = kernel_shape.dim[0];
            param.group = groups;

            //set shape
            auto last_call = GetRef<Call>(add_call);
            auto oshape = tvm::relay::backend::GetShape(last_call->checked_type());
            printf("[%d,%d,%d,%d]\n", oshape[0],oshape[1],oshape[2],oshape[3]);
            output_tensor->SetShape(GetShapeType(oshape));

            //quant output

            output_tensor->SetScale(docAct[(op_name_last + ":out").c_str()]["scale"][0].GetFloat());

            output_tensor->SetMax(docAct[(op_name_last + ":out").c_str()]["max_value"][0].GetFloat());
            output_tensor->SetMin(docAct[(op_name_last + ":out").c_str()]["min_value"][0].GetFloat());
			std::cout<<op_name + ":out"<<std::endl;
			if  (std::strcmp(docAct[(op_name_last + ":out").c_str()]["quantizer"].GetString(),"Symmetric")==0){
				output_tensor->SetQuantType(kQT_Sym);
                output_tensor->SetZeroPoint(docAct[(op_name_last + ":out").c_str()]["zero_point"].GetFloat());}
		    else{
				output_tensor->SetQuantType(kQT_Asym);
                output_tensor->SetZeroPoint(docAct[(op_name_last + ":out").c_str()]["zero_point"][0].GetFloat());}

            decltype(inputs) outputs{op_name + ":out"};

            //quant weight
            if (kernel_tensor->GetTensorType() == kTT_Const) {
            std::string kernel_quant_name;
            kernel_quant_name = kernel_quant_name + "nn.conv2d_transpose" + "_weight_" + std::to_string(layer_idx) + "_0:in";
            kernel_tensor->SetName(kernel_quant_name);
            std::string bias_quant_name;
            bias_quant_name = bias_quant_name + "add" + "_bias_" + std::to_string(layer_idx+1) + "_0:in";
            bias_tensor->SetName(bias_quant_name);

            std::vector<float> channel_scale(kernel_shape.dim[0]);
            std::vector<int> channel_zero_point(kernel_shape.dim[0]);
            std::vector<float> channel_max(kernel_shape.dim[0]);
            std::vector<float> channel_min(kernel_shape.dim[0]);

            const rapidjson::Value& scale = docWgt[(kernel_quant_name).c_str()]["scale"];
            const rapidjson::Value& zero_point = docWgt[(kernel_quant_name).c_str()]["zero_point"];
            const rapidjson::Value& max = docWgt[(kernel_quant_name).c_str()]["max_value"];
            const rapidjson::Value& min = docWgt[(kernel_quant_name).c_str()]["min_value"];
            if (scale.Size()>1) {

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
                printf("set conv_trans perchannel!!!!\n");
                param.quant_mode = kQM_PerChannel;
				if  (std::strcmp(docWgt[(kernel_quant_name).c_str()]["quantizer"].GetString(),"Symmetric")==0)
					kernel_tensor->SetQuantType(kQT_Sym);
		    	else
					kernel_tensor->SetQuantType(kQT_Asym);
            }
			else {
                kernel_tensor->SetScale(docWgt[(kernel_quant_name).c_str()]["scale"][0].GetFloat());
				kernel_tensor->SetMax(docWgt[(kernel_quant_name).c_str()]["max_value"][0].GetFloat());
				kernel_tensor->SetMin(docWgt[(kernel_quant_name).c_str()]["min_value"][0].GetFloat());
                param.quant_mode = kQM_PerTensor;
				if  (std::strcmp(docWgt[(kernel_quant_name).c_str()]["quantizer"].GetString(),"Symmetric")==0){
					kernel_tensor->SetQuantType(kQT_Sym);
					kernel_tensor->SetZeroPoint(docWgt[(kernel_quant_name).c_str()]["zero_point"].GetFloat());}
		    	else{
					kernel_tensor->SetQuantType(kQT_Asym);
					kernel_tensor->SetZeroPoint(docWgt[(kernel_quant_name).c_str()]["zero_point"][0].GetFloat());}

            }
            }
            if (doTiling)
            {
                std::string tiling_name;
                tiling_name = op_name_last;
                int output_tile_num = docTiling[(tiling_name).c_str()]["output_tile_num"].GetInt();
                int output_dim_num = docTiling[(tiling_name).c_str()]["output_dim_num"].GetInt();
                assert(output_dim_num == 4);
                int output_tile_dim = docTiling[(tiling_name).c_str()]["output_tile_dims"][0].GetInt();
                std::vector<TileTensorShape> tile_shape_list;
                printf("***********convtranspose************\n");
                printf("tiling_name: %s ", (tiling_name).c_str());
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
            //updata layer_idx for pattern
            layer_idx = layer_idx + 2;

            //CHECK_EQ(inputs.size(), func->num_inputs) << "wrong input arity: " << inputs.size();
            vis_tensor_map_[call] = outputs;

        }
		else if(pattern_name == "vis_vpu.batch_norm"){            // build input tensors
            printf("Into vis_vpu.batch_norm Node.\n");
            std::vector<std::string> inputs;
			int cnt = 0;
            for (const auto& arg : cn->args) {
				cnt++;
                CHECK(vis_tensor_map_.find(arg) != vis_tensor_map_.end());  // .at() gives bad message
                const auto& ts = vis_tensor_map_[arg];
                CHECK_EQ(ts.size(), 1) << "call arg is not expected be tuple";
                inputs.emplace_back(ts.back());
            }
            //create name

            std::string op_name = "batch_norm_" + std::to_string(layer_idx);
            std::string op_name_last = "add_" + std::to_string(layer_idx+1);
			std::cout<<"op_name = "<<op_name<<std::endl;

            //create node

            CHECK_EQ(inputs.size(), 3) << "number of inputs need to corrected";
            auto input_name = inputs.at(0);
            auto gamma_name = inputs.at(1);
            auto beta_name = inputs.back();

            TensorPtr input_tensor = graph->FindTensor(input_name);
            TensorPtr gamma_tensor = graph->FindTensor(gamma_name);
            TensorPtr beta_tensor = graph->FindTensor(beta_name);

            vis::megrez::DataType out_type = Get_quant_type(op_name_last);

            TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

            NodePtr node = graph->CreateNode(op_name, BN_OP);//

            node->SetInputTensor(0, input_tensor);
            node->SetOutputTensor(0, output_tensor);

            //parse attr

            std::vector<std::string> op_namelist = {"multiply", "add"};
            const CallNode* mul_call = GetRootCall(func->body.as<CallNode>(), 1, op_namelist);
            auto first_call = GetRef<Call>(mul_call);

            // config param

            BnOpPtr op_ptr = std::dynamic_pointer_cast<BnOp>(node->GetOp());//

            BnParam& param = op_ptr->GetParam();//

            int param_size = gamma_tensor->GetMemSize();

            param.scale.resize(param_size/4);
            param.b.resize(param_size/4);

			for(int i = 0 ; i < param_size/4; ++i)
				{
				param.var.push_back(1);
				param.mean.push_back(0);
				}

            std::memcpy(param.scale.data(), gamma_tensor->GetData(), param_size);
            std::memcpy(param.b.data(), beta_tensor->GetData(), param_size);


            //set shape
            std::vector<std::string> op_namelist_1 = {"add"};
            const CallNode* add_call = GetRootCall(func->body.as<CallNode>(), 0, op_namelist_1);
            auto last_call = GetRef<Call>(add_call);
            auto oshape = tvm::relay::backend::GetShape(last_call->checked_type());
            printf("[%d,%d,%d,%d]\n", oshape[0],oshape[1],oshape[2],oshape[3]);
            output_tensor->SetShape(GetShapeType(oshape));

            //quant output

            output_tensor->SetScale(docAct[(op_name_last + ":out").c_str()]["scale"][0].GetFloat());
            output_tensor->SetMax(docAct[(op_name_last + ":out").c_str()]["max_value"][0].GetFloat());
            output_tensor->SetMin(docAct[(op_name_last + ":out").c_str()]["min_value"][0].GetFloat());
			if  (std::strcmp(docAct[(op_name_last + ":out").c_str()]["quantizer"].GetString(),"Symmetric")==0) {
				output_tensor->SetQuantType(kQT_Sym);
				output_tensor->SetZeroPoint(docAct[(op_name_last + ":out").c_str()]["zero_point"].GetFloat());}

		    else {
				output_tensor->SetQuantType(kQT_Asym);
				output_tensor->SetZeroPoint(docAct[(op_name_last + ":out").c_str()]["zero_point"][0].GetFloat());}

            decltype(inputs) outputs{op_name + ":out"};

            //quant weight

            std::string gamma_tensor_name = "batch_norm_gamma_" + std::to_string(layer_idx) + "_0:in";

            gamma_tensor->SetName(gamma_tensor_name);
            std::string beta_tensor_name;
            beta_tensor_name = "add_beta_" + std::to_string(layer_idx+1) + "_0:in";
            beta_tensor->SetName(beta_tensor_name);

            //updata layer_idx for pattern
            layer_idx = layer_idx + 2;

            //CHECK_EQ(inputs.size(), func->num_inputs) << "wrong input arity: " << inputs.size();
            vis_tensor_map_[call] = outputs;

		}
        else if(pattern_name == "vis_vpu.mish"){   // build input tensors
            printf("Into vis_vpu.mish pattern Node.\n");
            std::vector<std::string> inputs;
			int cnt = 0;
            for (const auto& arg : cn->args) {
				cnt++;
                CHECK(vis_tensor_map_.find(arg) != vis_tensor_map_.end());  // .at() gives bad message
                const auto& ts = vis_tensor_map_[arg];
                CHECK_EQ(ts.size(), 1) << "call arg is not expected be tuple";
                inputs.emplace_back(ts.back());
            }
            //create name
            std::string op_name = "mish_" + std::to_string(layer_idx);
            std::string op_name_first = "exp_" + std::to_string(layer_idx);
            std::string op_name_last = "multiply_" + std::to_string(layer_idx+4);
			std::cout<<"op_name = "<<op_name<<std::endl;

            //create node

            CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";
            auto input_name = inputs.at(0);

            TensorPtr input_tensor = graph->FindTensor(input_name);

            vis::megrez::DataType out_type = Get_quant_type(op_name_last);
            TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

            NodePtr node = graph->CreateNode(op_name, MISH_OP);//

            node->SetInputTensor(0, input_tensor);
            node->SetOutputTensor(0, output_tensor);

            //set shape
            std::vector<std::string> op_namelist_1 = {"multiply"};
            const CallNode* muli_call = GetRootCall(func->body.as<CallNode>(), 0, op_namelist_1);
            auto last_call = GetRef<Call>(muli_call);
            auto oshape = tvm::relay::backend::GetShape(last_call->checked_type());
            //printf("[%d,%d,%d,%d]\n", oshape[0],oshape[1],oshape[2],oshape[3]);
            output_tensor->SetShape(GetShapeType(oshape));

            //quant output

            output_tensor->SetScale(docAct[(op_name_last + ":out").c_str()]["scale"][0].GetFloat());
            output_tensor->SetMax(docAct[(op_name_last + ":out").c_str()]["max_value"][0].GetFloat());
            output_tensor->SetMin(docAct[(op_name_last + ":out").c_str()]["min_value"][0].GetFloat());
			if  (std::strcmp(docAct[(op_name_last + ":out").c_str()]["quantizer"].GetString(),"Symmetric")==0) {
				output_tensor->SetQuantType(kQT_Sym);
				output_tensor->SetZeroPoint(docAct[(op_name_last + ":out").c_str()]["zero_point"].GetFloat());}

		    else {
				output_tensor->SetQuantType(kQT_Asym);
				output_tensor->SetZeroPoint(docAct[(op_name_last + ":out").c_str()]["zero_point"][0].GetFloat());}

            decltype(inputs) outputs{op_name + ":out"};

            layer_idx = layer_idx + 5;

            //CHECK_EQ(inputs.size(), func->num_inputs) << "wrong input arity: " << inputs.size();
            vis_tensor_map_[call] = outputs;

		}
        else if(pattern_name == "vis_vpu.gelu"){   // build input tensors
            printf("Into vis_vpu.gelu pattern Node.\n");
            std::vector<std::string> inputs;
			int cnt = 0;
            for (const auto& arg : cn->args) {
				cnt++;
                CHECK(vis_tensor_map_.find(arg) != vis_tensor_map_.end());  // .at() gives bad message
                const auto& ts = vis_tensor_map_[arg];
                CHECK_EQ(ts.size(), 1) << "call arg is not expected be tuple";
                inputs.emplace_back(ts.back());
            }
            //create name
            std::string op_name = "gelu_" + std::to_string(layer_idx);
            std::string op_name_first = "divide_" + std::to_string(layer_idx);
            std::string op_name_last = "multiply_" + std::to_string(layer_idx+4);
			std::cout<<"op_name = "<<op_name<<std::endl;

            //create node

            CHECK_EQ(inputs.size(), 1) << "number of inputs need to corrected";
            auto input_name = inputs.at(0);

            TensorPtr input_tensor = graph->FindTensor(input_name);

            vis::megrez::DataType out_type = Get_quant_type(op_name_last);
            TensorPtr output_tensor = graph->CreateTensor(op_name + ":out", kTT_Var, out_type);

            NodePtr node = graph->CreateNode(op_name, GELU_OP);//

            node->SetInputTensor(0, input_tensor);
            node->SetOutputTensor(0, output_tensor);

            //set shape
            std::vector<std::string> op_namelist_1 = {"multiply"};
            const CallNode* muli_call = GetRootCall(func->body.as<CallNode>(), 0, op_namelist_1);
            auto last_call = GetRef<Call>(muli_call);
            auto oshape = tvm::relay::backend::GetShape(last_call->checked_type());
            //printf("[%d,%d,%d,%d]\n", oshape[0],oshape[1],oshape[2],oshape[3]);
            output_tensor->SetShape(GetShapeType(oshape));

            //quant output

            output_tensor->SetScale(docAct[(op_name_last + ":out").c_str()]["scale"][0].GetFloat());
            output_tensor->SetMax(docAct[(op_name_last + ":out").c_str()]["max_value"][0].GetFloat());
            output_tensor->SetMin(docAct[(op_name_last + ":out").c_str()]["min_value"][0].GetFloat());
			if  (std::strcmp(docAct[(op_name_last + ":out").c_str()]["quantizer"].GetString(),"Symmetric")==0) {
				output_tensor->SetQuantType(kQT_Sym);
				output_tensor->SetZeroPoint(docAct[(op_name_last + ":out").c_str()]["zero_point"].GetFloat());}

		    else {
				output_tensor->SetQuantType(kQT_Asym);
				output_tensor->SetZeroPoint(docAct[(op_name_last + ":out").c_str()]["zero_point"][0].GetFloat());}

            decltype(inputs) outputs{op_name + ":out"};

            layer_idx = layer_idx + 5;

            //CHECK_EQ(inputs.size(), func->num_inputs) << "wrong input arity: " << inputs.size();
            vis_tensor_map_[call] = outputs;

		}
        else {
            printf("Into fault.\n");
            LOG(FATAL) << "Unknown composite function:" << pattern_name;
        }
    }

    else if (const auto* op_node = cn->op.as<OpNode>()) {
        printf("Into operation Node.\n");
        printf("op node is %s\n",op_node->name.c_str());
        // build input tensors
        std::vector<std::string> inputs;
        for (const auto& arg : cn->args) {
            CHECK(vis_tensor_map_.find(arg) != vis_tensor_map_.end());  // .at() gives bad message
            const auto& ts = vis_tensor_map_[arg];
            // Consider the concatenation situation (tuple) @2021/6/7
            for(size_t i=0; i<ts.size(); i++)
            {
                inputs.emplace_back(ts[i]);
            }
            // CHECK_EQ(ts.size(), 1) << "call arg is not expected be tuple";
            //inputs.emplace_back(ts.back());
        }
        //Consider the tuple situation @2021/6/7
        //CHECK_EQ(inputs.size(), op_node->num_inputs) << "wrong input arity: " << inputs.size();
        vis_tensor_map_[call] = BuildOp(call, inputs);

    } else {
        LOG(FATAL) << "Call op type not supported yet";
    }
}

void VisVpuCodegen::VisitExpr_(const TupleGetItemNode* op) {
    printf("Into tuplegetitemnode.\n");
    auto tgi = GetRef<TupleGetItem>(op);
    CHECK(vis_tensor_map_.find(tgi) == vis_tensor_map_.end());

    const auto it = vis_tensor_map_.find(op->tuple);
    CHECK(it != vis_tensor_map_.end());

    //set shape
    auto oshape = tvm::relay::backend::GetShape(tgi->checked_type());//todo
    TensorPtr output_tensor = graph->FindTensor(it->second.at(op->index));
    output_tensor->SetShape(GetShapeType(oshape));

    vis_tensor_map_[tgi] = {it->second.at(op->index)};
}

void VisVpuCodegen::VisitExpr_(const TupleNode* op) {
    printf("Into Tuple node.\n");
    auto tu = GetRef<Tuple>(op);
    CHECK(vis_tensor_map_.find(tu) == vis_tensor_map_.end());

    for (const auto& f : op->fields) {
        const auto it = vis_tensor_map_.find(f);
        CHECK(it != vis_tensor_map_.end());
        CHECK(it->second.size() == 1);  // tuple's field should not be a tuple
        vis_tensor_map_[tu].emplace_back(it->second.back());
    }
}

std::vector<std::string> VisVpuCodegen::BuildOp(const Call& call, const std::vector<std::string>& inputs) {
    auto outputs = Build(call, inputs);
    CHECK(!outputs.empty()) << "operator build failed: " << call;
    return outputs;
}

vis::megrez::ShapeType VisVpuCodegen::GetShapeType(const std::vector<int>& shape) {
    if (shape.size() == 1) {

        vis::megrez::ShapeType shapetype = vis::megrez::ShapeType();
        shapetype.dim_num = 1;
        shapetype.dim[0] = shape[0];

        return shapetype;
    } else if (shape.size() == 2) {

        vis::megrez::ShapeType shapetype = vis::megrez::ShapeType();
        shapetype.dim_num = 2;
        shapetype.dim[0] = shape[0];
        shapetype.dim[1] = shape[1];

        return shapetype;
    } else if (shape.size() == 3) {

        vis::megrez::ShapeType shapetype = vis::megrez::ShapeType();
        shapetype.dim_num = 3;
        shapetype.dim[0] = shape[0];
        shapetype.dim[1] = shape[1];
        shapetype.dim[2] = shape[2];
        return shapetype;
    } else if (shape.size() == 4) {
        return vis::megrez::ShapeType(shape[0], shape[1], shape[2], shape[3]);
    } else if (shape.size() == 5) {

        vis::megrez::ShapeType shapetype = vis::megrez::ShapeType();
        shapetype.dim_num = 5;
        shapetype.dim[0] = shape[0];
        shapetype.dim[1] = shape[1];
        shapetype.dim[2] = shape[2];
        shapetype.dim[3] = shape[3];
        shapetype.dim[4] = shape[4];
        return shapetype;
    } else if (shape.size() == 0) {

        vis::megrez::ShapeType shapetype = vis::megrez::ShapeType();
        shapetype.dim_num = 1;
        shapetype.dim[0] = 1;
        return shapetype;
    }else {
        LOG(FATAL) << "Unsupport shape" << std::to_string(shape.size());
    }
}

vis::megrez::DataType VisVpuCodegen::Tvm2VisVpu(const DataType& d) {
    std::unordered_map<DataType, vis::megrez::DataType> tvm2visvpu = {

        {DataType::Float(16), vis::megrez::DataType::kDT_FP16},
        {DataType::Float(32), vis::megrez::DataType::kDT_FP32},
        {DataType::Int(8), vis::megrez::DataType::kDT_Int8},
        {DataType::Int(16), vis::megrez::DataType::kDT_Int16},
        {DataType::UInt(8), vis::megrez::DataType::kDT_Uint8},
        {DataType::UInt(16), vis::megrez::DataType::kDT_Uint16},
        {DataType::Int(32), vis::megrez::DataType::kDT_Int32},
        {DataType::Int(64), vis::megrez::DataType::kDT_Int32}

    };
    auto it = tvm2visvpu.find(d);
    CHECK(it != tvm2visvpu.end()) << "vis unsupported data type: " << d;  // at() gives bad message
    return it->second;
}

tvm::runtime::Module VisVpuCompile(const ObjectRef& ref) {
    CHECK(ref->IsInstance<FunctionNode>()) << "The input ref is expected to be a VisVpu function.";
    Function func = Downcast<Function>(ref);
    std::string func_name = backend::GetExtSymbol(func);

    VisVpuCodegen codegen(func_name, func);
    codegen.BuildGraph();

    auto consts = codegen.GetConstParams();

    //fixme:: buffer is loadable, need update this code
    auto n = make_object<::tvm::runtime::contrib::VISModule>(func_name, consts);
    return tvm::runtime::Module(n);
}

TVM_REGISTER_GLOBAL("relay.ext.vis_vpu").set_body_typed(VisVpuCompile);

TVM_REGISTER_GLOBAL("relay.ext.vis_vpu.constant_updater")
    .set_body_typed([](const Expr& expr, const std::string& symbol) {
        // storing tensor consts in VisVpu binary graph
        return Map<String, tvm::runtime::NDArray>{};
    });

TVM_REGISTER_NODE_TYPE(VisVpuConfigNode);

TVM_REGISTER_PASS_CONFIG_OPTION("relay.ext.vis_vpu.options", VisVpuConfig);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
