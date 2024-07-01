/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
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


#include "holoscan/core/gxf/gxf_execution_context.hpp"

#include <memory>
#include <utility>

#include "holoscan/core/gxf/gxf_operator.hpp"

namespace holoscan::gxf {

GXFExecutionContext::GXFExecutionContext(gxf_context_t context, Operator* op) {
  gxf_input_context_ = std::make_shared<GXFInputContext>(this, op);
  gxf_output_context_ = std::make_shared<GXFOutputContext>(this, op);

  context_ = context;
  input_context_ = gxf_input_context_.get();
  output_context_ = gxf_output_context_.get();
}

GXFExecutionContext::GXFExecutionContext(gxf_context_t context,
                                         std::shared_ptr<GXFInputContext> gxf_input_context,
                                         std::shared_ptr<GXFOutputContext> gxf_output_context)
    : gxf_input_context_(std::move(gxf_input_context)),
      gxf_output_context_(std::move(gxf_output_context)) {
  context_ = context;
}

}  // namespace holoscan::gxf
