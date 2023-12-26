// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <ctime>
#include <vector>
#include <memory>
#include <set>
#include <string>

#include "core/framework/execution_provider.h"

#include "core/providers/iree/iree_ep_runtime.h"

namespace onnxruntime {

// Logical device representation.
class IREEExecutionProvider : public IExecutionProvider {
 public:
  explicit IREEExecutionProvider(const ProviderOptions& info);
  ~IREEExecutionProvider() override;

  // Failable initialization activities.
  common::Status Initialize();

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(const onnxruntime::GraphViewer& graph,
                                                                const IKernelLookup& /*kernel_lookup*/) const override;

  int GetDeviceId() const override { return 0; }

  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;
  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

 private:
  void CreateKernelRegistry();
  NodeComputeInfo CreateNodeComputeFunc(std::string entrypoint_name, std::shared_ptr<iree_ep_rt::Session> session);

  ProviderOptions info_;
  std::shared_ptr<KernelRegistry> registry_;

  // TODO: We may want to make the instance into something that is shared across EP instances.
  // The critical thing is that we don't want to be having multiples of the underlying VM instance
  // or HAL devices contained herein. This usually requires some form of process scoping in systems
  // like this.
  std::shared_ptr<iree_ep_rt::Instance> rt_instance_;
};

}  // namespace onnxruntime
