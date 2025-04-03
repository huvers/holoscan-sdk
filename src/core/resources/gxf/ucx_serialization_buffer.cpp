/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "holoscan/core/resources/gxf/ucx_serialization_buffer.hpp"

#include <stdlib.h>  // setenv

#include <algorithm>
#include <memory>
#include <string>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"

namespace holoscan {

// Note: UcxSerializationBuffer does not inherit from SerializationBuffer
UcxSerializationBuffer::UcxSerializationBuffer(const std::string& name,
                                               nvidia::gxf::UcxSerializationBuffer* component)
    : gxf::GXFResource(name, component) {
  auto maybe_buffer_size = component->getParameter<size_t>("buffer_size");
  if (!maybe_buffer_size) { throw std::runtime_error("Failed to get maybe_buffer_size"); }
  buffer_size_ = maybe_buffer_size.value();

  auto maybe_allocator =
      component->getParameter<nvidia::gxf::Handle<nvidia::gxf::Allocator>>("allocator");
  if (!maybe_allocator) { throw std::runtime_error("Failed to get allocator"); }
  auto allocator_handle = maybe_allocator.value();
  allocator_ =
      std::make_shared<Allocator>(std::string{allocator_handle->name()}, allocator_handle.get());
}

void UcxSerializationBuffer::setup(ComponentSpec& spec) {
  std::string buffer_env_name{"HOLOSCAN_UCX_SERIALIZATION_BUFFER_SIZE"};
  const char* env_value = std::getenv(buffer_env_name.c_str());
  size_t default_buffer_size = 0;
  if (env_value) {
    try {
      default_buffer_size = std::stoull(env_value);
      HOLOSCAN_LOG_DEBUG("UcxSerializationBuffer: setting buffer size to {}", default_buffer_size);

      // Need to set corresponding underlying UCX environment variables as well or an error
      // such as the following may be seen at run time
      //     ucp_am.c:758  Fatal: RTS is too big XXXX, max YYYY
      setenv("UCX_TCP_RX_SEG_SIZE", env_value, 0);
      setenv("UCX_TCP_TX_SEG_SIZE", env_value, 0);
    } catch (std::exception& e) {
      HOLOSCAN_LOG_WARN(
          "Unable to interpret environment variable '{}': '{}'", buffer_env_name, e.what());
    }
  } else {
    default_buffer_size = kDefaultUcxSerializationBufferSize;
  }

  spec.param(allocator_, "allocator", "Memory allocator", "Memory allocator for tensor components");
  spec.param(buffer_size_,
             "buffer_size",
             "Buffer Size",
             "Size of the buffer in bytes (7168 by default unless "
             "HOLOSCAN_UCX_SERIALIZATION_BUFFER_SIZE is defined)",
             default_buffer_size);
}

nvidia::gxf::UcxSerializationBuffer* UcxSerializationBuffer::get() const {
  return static_cast<nvidia::gxf::UcxSerializationBuffer*>(gxf_cptr_);
}

void UcxSerializationBuffer::initialize() {
  // Set up prerequisite parameters before calling GXFResource::initialize()
  auto frag = fragment();

  // Find if there is an argument for 'allocator'
  auto has_allocator = std::find_if(
      args().begin(), args().end(), [](const auto& arg) { return (arg.name() == "allocator"); });
  // Create an UnboundedAllocator if no allocator was provided
  if (has_allocator == args().end()) {
    auto allocator = frag->make_resource<UnboundedAllocator>("ucx_serialization_buffer_allocator");
    allocator->gxf_cname(allocator->name().c_str());
    if (gxf_eid_ != 0) { allocator->gxf_eid(gxf_eid_); }
    add_arg(Arg("allocator") = allocator);
  } else {
    // must set the gxf_eid for the provided allocator or GXF parameter registration will fail
    auto allocator_arg = *has_allocator;
    auto allocator = std::any_cast<std::shared_ptr<Resource>>(allocator_arg.value());
    auto gxf_allocator_resource = std::dynamic_pointer_cast<gxf::GXFResource>(allocator);
    if (gxf_eid_ != 0 && gxf_allocator_resource->gxf_eid() == 0) {
      HOLOSCAN_LOG_TRACE("allocator '{}': setting gxf_eid({}) from UcxSerializationBuffer '{}'",
                         allocator->name(),
                         gxf_eid_,
                         name());
      gxf_allocator_resource->gxf_eid(gxf_eid_);
    }
  }
  GXFResource::initialize();
}

}  // namespace holoscan
