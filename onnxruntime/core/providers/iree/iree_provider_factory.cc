// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "./iree_execution_provider.h"
#include "./iree_provider_factory_creator.h"

#include "core/common/common.h"

using namespace onnxruntime;

namespace onnxruntime {

struct IREEProviderFactory : IExecutionProviderFactory {
  IREEProviderFactory(const ProviderOptions& info) : info_(info) {}
  ~IREEProviderFactory() = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  ProviderOptions info_;
};

std::unique_ptr<IExecutionProvider> IREEProviderFactory::CreateProvider() {
  return std::make_unique<IREEExecutionProvider>(info_);
}

std::shared_ptr<IExecutionProviderFactory> IREEProviderFactoryCreator::Create(
    const ProviderOptions& provider_options) {
  ORT_UNUSED_PARAMETER(provider_options);
  return std::make_shared<IREEProviderFactory>(provider_options);
  return nullptr;
}

}  // namespace onnxruntime
