// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/rocm/miopen_common.h"
#include "core/providers/rocm/nn/conv.h"
#include "core/providers/cpu/nn/conv_transpose_attributes.h"

namespace onnxruntime {
namespace rocm {

template <typename T, bool NHWC>
class ConvTranspose : public RocmKernel {
 public:
  using HipT = typename ToHipType<T>::MappedType;

  ConvTranspose(const OpKernelInfo& info) : RocmKernel(info), conv_transpose_attrs_(info) {
    is_nhwc_domain_ = info.node().Domain() == kMSInternalNHWCDomain;
  };
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 bool& is_packed, PrePackedWeights* prepacked_weights) override;
  Status ComputeInternal(OpKernelContext* context) const override;
  Status DoConvTranspose(OpKernelContext* context, bool dynamic_padding) const;

 private:
  ConvTransposeAttributes conv_transpose_attrs_;

  mutable MiopenConvState<miopenConvAlgoPerf_t> s_;
  std::unique_ptr<Tensor> W_;

  bool is_nhwc_domain_;         // prepack is only needed for the Conv in kMSInternalNHWCDomain
  bool is_fused_node_ = false;  // ensures the node is fused although the session option is not set
  bool W_already_nhwc = false;  // In case NHWC == true and Conv is not in kMSInternalNHWCDomain

 protected:
  inline IAllocatorUniquePtr<void> GetWorkSpace(onnxruntime::Stream* stream) const {
    return GetScratchBuffer<void>(s_.workspace_bytes, stream);
  }

  Status UpdateState(OpKernelContext* context, bool bias_expected) const;
};

}  // namespace rocm
}  // namespace onnxruntime
