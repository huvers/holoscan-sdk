/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_CONDITIONS_GXF_DOWNSTREAM_AFFORDABLE_HPP
#define HOLOSCAN_CORE_CONDITIONS_GXF_DOWNSTREAM_AFFORDABLE_HPP

#include <memory>

#include <gxf/std/scheduling_terms.hpp>

#include "../../gxf/gxf_condition.hpp"
#include "../../resources/gxf/transmitter.hpp"

namespace holoscan {

/**
 * @brief Condition class that allows an operator to execute only when there is space in any
 * downstream operator's receiver queues for a specified number of messages produced by a given
 * output port.
 *
 * This condition applies to a specific output port of the operator as determined by setting the
 * "transmitter" argument.
 *
 * This condition can also be set via the `Operator::setup` method using `IOSpec::condition` with
 * `ConditionType::kDownstreamMessageAffordable`. In that case, the transmitter is already known
 * from the port corresponding to the `IOSpec` object, so the "transmitter" argument is unnecessary.
 *
 * ==Parameters==
 *
 * - **min_size** (uint64_t): The minimum number of messages that there must be space available for
 * in the front stage of the double-buffer receiver queues of all receivers connected to the
 * specified transmitter.
 * - **transmitter** (std::string): The transmitter that should check for space in the queue of all
 * of its connected receivers. This should be specified by the name of the operator's output port
 * the condition will apply to. The Holoscan SDK will then automatically replace the port name with
 * the actual transmitter object at application run time.
 */
class DownstreamMessageAffordableCondition : public gxf::GXFCondition {
 public:
  HOLOSCAN_CONDITION_FORWARD_ARGS_SUPER(DownstreamMessageAffordableCondition, GXFCondition)
  DownstreamMessageAffordableCondition() = default;
  explicit DownstreamMessageAffordableCondition(size_t min_size) : min_size_(min_size) {}

  const char* gxf_typename() const override {
    return "nvidia::gxf::DownstreamReceptiveSchedulingTerm";
  }

  void setup(ComponentSpec& spec) override;

  // TODO(GXF4):   Expected<void> setTransmitter(Handle<Transmitter> value)
  void transmitter(std::shared_ptr<Transmitter> transmitter) { transmitter_ = transmitter; }
  std::shared_ptr<Transmitter> transmitter() { return transmitter_; }

  void min_size(uint64_t min_size);
  uint64_t min_size() { return min_size_; }

  void initialize() override { GXFCondition::initialize(); }

  nvidia::gxf::DownstreamReceptiveSchedulingTerm* get() const;

 private:
  Parameter<std::shared_ptr<Transmitter>> transmitter_;
  Parameter<uint64_t> min_size_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_CONDITIONS_GXF_DOWNSTREAM_AFFORDABLE_HPP */
