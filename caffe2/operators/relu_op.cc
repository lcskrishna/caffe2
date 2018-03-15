/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "caffe2/operators/relu_op.h"

#include "caffe2/utils/math.h"

#include <stdio.h>
#include <vector>
#include <fstream>

#include "dump_layers.h"

namespace caffe2 {

template <>
bool ReluOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);

#ifdef CAFFE2_USE_ACCELERATE
  const float zero = 0.0f;
  vDSP_vthres(X.data<float>(), 1, &zero, Y->mutable_data<float>(), 1, X.size());
#else
  EigenVectorMap<float>(Y->mutable_data<float>(), X.size()) =
      ConstEigenVectorMap<float>(X.data<float>(), X.size()).cwiseMax(0.f);
#endif
  /* Naive implementation
  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();
  for (int i = 0; i < X.size(); ++i) {
    Ydata[i] = std::max(Xdata[i], 0.f);
  }
  */  

#if ENABLE_DUMP_LAYERS
  std::cout << "INFO: Relu Operator called" << std::endl;

  /*
  //Get layer number.
  int layer_num = get_layer_count();

  //calculation of the input dims.
  vector<long int> in_tensor_dims = X.dims();
  long input_count = 1;
  for(int i=0; i < in_tensor_dims.size(); i++) {
       input_count *= in_tensor_dims[i];
  }

  //calculation of the output dims.
  vector<long int> out_tensor_dims = Y->dims();
  long output_count = 1;
  for(int i=0; i < out_tensor_dims.size(); i++) {
      output_count *= out_tensor_dims[i];
  }

  //Dump input to relu layer.
  std::string input_file_name = "dump/input_relu_layer_" + std::to_string(layer_num);
  FILE * fs  = fopen(input_file_name.c_str(), "wb");
  for(int i=0; i < input_count; i++) {
      float val = X.data<float>()[i];
      fwrite(&val, sizeof(float), 1, fs);
  }
  fclose(fs);

  //Dump output layer to convolution layer.
  std::string output_file_name = "dump/output_relu_layer_" + std::to_string(layer_num);
  FILE * fp = fopen(output_file_name.c_str(), "wb");
  for(int i=0; i < output_count; i++) {
      float val = Y->data<float>()[i];
      fwrite(&val, sizeof(float), 1, fp);
  }
  fclose(fp);
  */

  increment_layer_count();
#endif

  return true;
}

template <>
bool ReluGradientOp<float, CPUContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  CAFFE_ENFORCE_EQ(dY.size(), Y.size());
  dX->ResizeLike(Y);

  const float* Ydata = Y.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
  // TODO: proper vectorization with Eigen
  EigenVectorArrayMap<float> dXvec(dXdata, dX->size());
  ConstEigenVectorArrayMap<float> Yvec(Ydata, Y.size());
  ConstEigenVectorArrayMap<float> dYvec(dYdata, dY.size());
  dXvec = dYvec * Yvec.cwiseSign();
  /* Previous implementation
  for (int i = 0; i < Y.size(); ++i) {
    dXdata[i] = Ydata[i] > 0 ? dYdata[i] : 0;
  }
  */
  return true;
}

namespace {
OpSchema::Cost CostInferenceForRelu(
    const OperatorDef& def,
    const vector<TensorShape>& in) {
  struct OpSchema::Cost cost = PointwiseCostInference<0>(def, in);
  if (def.input(0) == def.output(0)) {
    cost.bytes_moved = 0;
  }
  cost.params_bytes = 0;
  return cost;
}
} // namespace

REGISTER_CPU_OPERATOR(Relu, ReluOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(ReluGradient, ReluGradientOp<float, CPUContext>);

// Input: X, output: Y
OPERATOR_SCHEMA(Relu)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .CostInferenceFunction(CostInferenceForRelu)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Relu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
the tensor elementwise.
)DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D input tensor");

// Input: Y, dY, output: dX
OPERATOR_SCHEMA(ReluGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{1, 0}})
    .SetDoc(R"DOC(
ReluGradient takes both Y and dY and uses this to update dX according to the
chain rule and derivatives of the rectified linear function.
)DOC");

class GetReluGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient",
        "",
        vector<string>{O(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Relu, GetReluGradient);
REGISTER_GRADIENT(ReluFp16, GetReluGradient);

}  // namespace caffe2
