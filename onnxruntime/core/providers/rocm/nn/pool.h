// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/nn/pool_base.h"
#include "core/providers/rocm/miopen_common.h"
#include "core/providers/rocm/rocm_kernel.h"

namespace onnxruntime {
namespace rocm {

template <typename T, typename PoolType, bool NHWC>
class Pool : public RocmKernel, public PoolBase {
 public:
  Pool(const OpKernelInfo& info) : RocmKernel(info), PoolBase(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T, bool Layout>
class Pool<T, MaxPool<8>, Layout> final : public Pool<T, MaxPool<1>, Layout> {
 public:
  explicit Pool(const OpKernelInfo& info) : Pool<T, MaxPool<1>, Layout>(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};


}  // namespace rocm
}  // namespace onnxruntime
