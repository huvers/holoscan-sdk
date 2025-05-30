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

#ifndef HOLOSCAN_OPERATORS_ASYNC_PING_RX_HPP
#define HOLOSCAN_OPERATORS_ASYNC_PING_RX_HPP

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>

#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

/**
 * @brief Simple asynchronous receiver operator.
 *
 * ==Named Inputs==
 *
 * - **in** : any
 *   - A received value.
 *
 * ==Parameters==
 *
 * - **delay**: Ping delay in ms. Optional (default: `10L`)
 * - **async_condition**: AsynchronousCondition adding async support to the operator.
 *   Optional (default: `nullptr`)
 */
class AsyncPingRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AsyncPingRxOp)

  AsyncPingRxOp() = default;

  void setup(OperatorSpec& spec) override;
  void start() override;
  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override;
  void stop() override;

  void async_ping();

 private:
  Parameter<int64_t> delay_;

  // internal state
  std::atomic<bool> should_stop_{false};
  std::thread async_thread_;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_ASYNC_PING_RX_HPP */
