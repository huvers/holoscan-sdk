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

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_ENTITY_SERIALIZER_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_ENTITY_SERIALIZER_HPP

#include <memory>
#include <vector>

#include <gxf/ucx/ucx_entity_serializer.hpp>

#include "../../gxf/gxf_resource.hpp"

namespace holoscan {

/**
 * @brief UCX-based entity serializer.
 *
 * Used by UcxReceiver and UcxTransmitter to serialize and deserialize entities, respectively.
 *
 * ==Parameters==
 *
 * - **component_serializers** (std::vector<std::shared_ptr<holoscan::Resource>>): The component
 * serializers available for serialization/deserialization of components in the Entity. By
 * default, Holoscan uses both `UcxComponentSerializer` and `UcxHoloscanComponentSerializer`.
 * - **verbose_warning** (bool): If true, more verbose warnings are logged by the serializer.
 */
class UcxEntitySerializer : public gxf::GXFResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(UcxEntitySerializer, GXFResource)
  UcxEntitySerializer() = default;

  const char* gxf_typename() const override { return "nvidia::gxf::UcxEntitySerializer"; }

  void setup(ComponentSpec& spec) override;

  void initialize() override;

  nvidia::gxf::UcxEntitySerializer* get() const;

 private:
  Parameter<std::vector<std::shared_ptr<holoscan::Resource>>> component_serializers_;
  Parameter<bool> verbose_warning_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_ENTITY_SERIALIZER_HPP */
