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

#include <gtest/gtest.h>
#include <gxf/core/gxf.h>

#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../utils.hpp"
#include "common/assert.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/condition.hpp"
#include "holoscan/core/conditions/gxf/asynchronous.hpp"
#include "holoscan/core/conditions/gxf/boolean.hpp"
#include "holoscan/core/conditions/gxf/count.hpp"
#include "holoscan/core/conditions/gxf/cuda_buffer_available.hpp"
#include "holoscan/core/conditions/gxf/cuda_event.hpp"
#include "holoscan/core/conditions/gxf/cuda_stream.hpp"
#include "holoscan/core/conditions/gxf/downstream_affordable.hpp"
#include "holoscan/core/conditions/gxf/expiring_message.hpp"
#include "holoscan/core/conditions/gxf/memory_available.hpp"
#include "holoscan/core/conditions/gxf/message_available.hpp"
#include "holoscan/core/conditions/gxf/multi_message_available.hpp"
#include "holoscan/core/conditions/gxf/multi_message_available_timeout.hpp"
#include "holoscan/core/conditions/gxf/periodic.hpp"
#include "holoscan/core/config.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/resources/gxf/unbounded_allocator.hpp"

using namespace std::string_literals;

namespace holoscan {

using ConditionClassesWithGXFContext = TestWithGXFContext;

TEST(ConditionClasses, TestAsynchronousCondition) {
  Fragment F;
  const std::string name{"async-condition"};
  auto condition = F.make_condition<AsynchronousCondition>(name);
  EXPECT_EQ(condition->name(), name);
  EXPECT_EQ(typeid(condition), typeid(std::make_shared<AsynchronousCondition>()));
  EXPECT_EQ(std::string(condition->gxf_typename()), "nvidia::gxf::AsynchronousSchedulingTerm"s);
  EXPECT_TRUE(condition->description().find("name: " + name) != std::string::npos);
}

TEST(ConditionClasses, TestAsynchronousConditionEventState) {
  Fragment F;
  const std::string name{"async-condition"};
  auto condition = F.make_condition<AsynchronousCondition>(name);
  EXPECT_EQ(condition->event_state(), holoscan::AsynchronousEventState::READY);

  condition->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);
  EXPECT_EQ(condition->event_state(), holoscan::AsynchronousEventState::EVENT_WAITING);
}

TEST(ConditionClasses, TestBooleanCondition) {
  Fragment F;
  const std::string name{"boolean-condition"};
  auto condition = F.make_condition<BooleanCondition>(name, Arg{"enable_tick", true});
  EXPECT_EQ(condition->name(), name);
  EXPECT_EQ(typeid(condition), typeid(std::make_shared<BooleanCondition>(true)));
  EXPECT_EQ(std::string(condition->gxf_typename()), "nvidia::gxf::BooleanSchedulingTerm"s);
  EXPECT_TRUE(condition->description().find("name: " + name) != std::string::npos);
}

TEST(ConditionClasses, TestBooleanConditionEnabledMethods) {
  Fragment F;
  const std::string name{"boolean-condition"};
  auto condition = F.make_condition<BooleanCondition>(name, Arg{"enable_tick", true});

  // can't call check_tick_enabled before it has been set
  // (make_condition does not call initialize() on the condition)
  try {
    condition->check_tick_enabled();
  } catch (const std::runtime_error& e) {
    // and this tests that it has the correct message
    EXPECT_TRUE(std::string(e.what()).find("'enable_tick' is not set") != std::string::npos);
  }

  // check disable and enable
  condition->disable_tick();
  EXPECT_EQ(condition->check_tick_enabled(), false);
  condition->enable_tick();
  EXPECT_EQ(condition->check_tick_enabled(), true);
}

TEST(ConditionClasses, TestBooleanConditionDefaultConstructor) {
  Fragment F;
  auto condition = F.make_condition<BooleanCondition>();
}

TEST(ConditionClasses, TestCountCondition) {
  Fragment F;
  const std::string name{"count-condition"};
  auto condition = F.make_condition<CountCondition>(name, Arg{"count", 100});
  EXPECT_EQ(condition->name(), name);
  EXPECT_EQ(typeid(condition), typeid(std::make_shared<CountCondition>(100)));
}

TEST(ConditionClasses, TestCountConditionDefaultConstructor) {
  Fragment F;
  auto condition = F.make_condition<CountCondition>();
}

TEST(ConditionClasses, TestCountConditionMethods) {
  Fragment F;
  const std::string name{"count-condition"};
  auto condition = F.make_condition<CountCondition>(name, Arg{"count", 100});

  condition->count(20);
  EXPECT_EQ(condition->count(), 20);

  // can access methods of GXFComponent
  EXPECT_EQ(std::string(condition->gxf_typename()), "nvidia::gxf::CountSchedulingTerm"s);
  EXPECT_TRUE(condition->description().find("name: " + name) != std::string::npos);
}

TEST(ConditionClasses, TestCountConditionGXFComponentMethods) {
  Fragment F;
  const std::string name{"count-condition"};
  auto condition = F.make_condition<CountCondition>(name, Arg{"count", 100});

  // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
  auto gxf_typename = condition->gxf_typename();
  auto context = condition->gxf_context();
  auto cid = condition->gxf_cid();
  auto eid = condition->gxf_eid();
  // NOLINTEND(clang-analyzer-deadcode.DeadStores)
}

TEST_F(ConditionClassesWithGXFContext, TestCountConditionInitializeWithoutSpec) {
  CountCondition count{10};
  count.fragment(&F);
  // TODO(unknown): avoid segfault if initialize is called before the fragment is assigned

  // test that an error is logged if initialize is called before a spec as assigned
  testing::internal::CaptureStderr();
  count.initialize();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") != std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("No component spec") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST_F(ConditionClassesWithGXFContext, TestCountConditionInitializeWithUnrecognizedArg) {
  auto condition = F.make_condition<CountCondition>(Arg{"count", 100}, Arg("undefined_arg", 5.0));

  // test that an warning is logged if an unknown argument is provided
  testing::internal::CaptureStderr();
  condition->initialize();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("warning") != std::string::npos) << "=== LOG ===\n"
                                                               << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("'undefined_arg' not found in spec_.params") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(ConditionClasses, TestDownstreamMessageAffordableCondition) {
  Fragment F;
  const std::string name{"downstream-message-affordable-condition"};
  ArgList arglist{
      Arg{"min_size", 1L},
  };
  auto condition = F.make_condition<DownstreamMessageAffordableCondition>(name, arglist);
  EXPECT_EQ(condition->name(), name);
  EXPECT_EQ(typeid(condition),
            typeid(std::make_shared<DownstreamMessageAffordableCondition>(arglist)));
  EXPECT_EQ(std::string(condition->gxf_typename()),
            "nvidia::gxf::DownstreamReceptiveSchedulingTerm"s);
  EXPECT_TRUE(condition->description().find("name: " + name) != std::string::npos);
}

TEST(ConditionClasses, TestDownstreamMessageAffordableConditionDefaultConstructor) {
  Fragment F;
  auto condition = F.make_condition<DownstreamMessageAffordableCondition>();
}

TEST(ConditionClasses, TestDownstreamMessageAffordableConditionSizeMethod) {
  Fragment F;
  const std::string name{"downstream-message-affordable-condition"};
  ArgList arglist{
      Arg{"min_size", 1L},
  };
  auto condition = F.make_condition<DownstreamMessageAffordableCondition>(name, arglist);
  condition->min_size(16);
  EXPECT_EQ(condition->min_size(), 16);
}

TEST(ConditionClasses, TestMemoryAvailableCondition) {
  Fragment F;
  const std::string name{"memory-available-condition"};
  ArgList arglist{Arg{"allocator", F.make_resource<UnboundedAllocator>()},
                  Arg{"min_bytes", static_cast<uint64_t>(1'000'000)}};
  auto condition = F.make_condition<MemoryAvailableCondition>(name, arglist);
  EXPECT_EQ(condition->name(), name);
  EXPECT_EQ(typeid(condition), typeid(std::make_shared<MemoryAvailableCondition>(arglist)));
  EXPECT_EQ(std::string(condition->gxf_typename()), "nvidia::gxf::MemoryAvailableSchedulingTerm"s);
  EXPECT_TRUE(condition->description().find("name: " + name) != std::string::npos);
}

TEST(ConditionClasses, TestMemoryAvailableConditionMinBlocks) {
  Fragment F;
  const std::string name{"memory-available-condition"};
  ArgList arglist{Arg{"allocator", F.make_resource<UnboundedAllocator>()},
                  Arg{"min_blocks", static_cast<uint64_t>(2)}};
  auto condition = F.make_condition<MemoryAvailableCondition>(name, arglist);
  EXPECT_EQ(condition->name(), name);
  EXPECT_EQ(typeid(condition), typeid(std::make_shared<MemoryAvailableCondition>(arglist)));
  EXPECT_EQ(std::string(condition->gxf_typename()), "nvidia::gxf::MemoryAvailableSchedulingTerm"s);
  EXPECT_TRUE(condition->description().find("name: " + name) != std::string::npos);
}

TEST(ConditionClasses, TestMemoryAvailableConditionDefaultConstructor) {
  Fragment F;
  auto condition = F.make_condition<MemoryAvailableCondition>();
}

TEST(ConditionClasses, TestMessageAvailableCondition) {
  Fragment F;
  const std::string name{"message-available-condition"};
  ArgList arglist{Arg{"min_size", 1L}, Arg{"front_stage_max_size", 2L}};
  auto condition = F.make_condition<MessageAvailableCondition>(name, arglist);
  EXPECT_EQ(condition->name(), name);
  EXPECT_EQ(typeid(condition), typeid(std::make_shared<MessageAvailableCondition>(arglist)));
  EXPECT_EQ(std::string(condition->gxf_typename()), "nvidia::gxf::MessageAvailableSchedulingTerm"s);
  EXPECT_TRUE(condition->description().find("name: " + name) != std::string::npos);
}

TEST(ConditionClasses, TestMessageAvailableConditionDefaultConstructor) {
  Fragment F;
  auto condition = F.make_condition<MessageAvailableCondition>();
}

TEST(ConditionClasses, TestMessageAvailableConditionSizeMethods) {
  Fragment F;
  const std::string name{"message-available-condition"};
  ArgList arglist{Arg{"min_size", 1L}, Arg{"front_stage_max_size", 2L}};
  auto condition = F.make_condition<MessageAvailableCondition>(name, arglist);

  condition->min_size(3);
  EXPECT_EQ(condition->min_size(), 3);

  condition->front_stage_max_size(5);
  EXPECT_EQ(condition->front_stage_max_size(), 5);
}

TEST(ConditionClasses, TestExpiringMessageAvailableCondition) {
  Fragment F;
  const std::string name{"expiring-message-available-condition"};
  ArgList arglist{Arg{"min_size", 1L}, Arg{"front_stage_max_size", 2L}};
  auto condition = F.make_condition<ExpiringMessageAvailableCondition>(name, arglist);
  EXPECT_EQ(condition->name(), name);
  EXPECT_EQ(typeid(condition),
            typeid(std::make_shared<ExpiringMessageAvailableCondition>(arglist)));
  EXPECT_EQ(std::string(condition->gxf_typename()),
            "nvidia::gxf::ExpiringMessageAvailableSchedulingTerm"s);
  EXPECT_TRUE(condition->description().find("name: " + name) != std::string::npos);
}

TEST(ConditionClasses, TestExpiringMessageAvailableConditionDefaultConstructor) {
  Fragment F;
  auto condition = F.make_condition<ExpiringMessageAvailableCondition>();
}

TEST(ConditionClasses, TestMultiMessageAvailableConditionSumOfAll) {
  // Test supplying sampling_mode argument as an enum
  Fragment F;
  const std::string name{"multi-message-available"};
  ArgList arglist{Arg{"sampling_mode", MultiMessageAvailableCondition::SamplingMode::kSumOfAll},
                  Arg{"min_sum", static_cast<size_t>(4)}};
  auto condition = F.make_condition<MultiMessageAvailableCondition>(name, arglist);
  EXPECT_EQ(condition->name(), name);
  EXPECT_EQ(typeid(condition), typeid(std::make_shared<MultiMessageAvailableCondition>(arglist)));
  EXPECT_EQ(std::string(condition->gxf_typename()),
            "nvidia::gxf::MultiMessageAvailableSchedulingTerm"s);
  EXPECT_TRUE(condition->description().find("name: " + name) != std::string::npos);

  // verify no error logged about failure to convert sampling_mode to YAML::Node was logged
  testing::internal::CaptureStderr();

  condition->initialize();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Unable to convert argument type") == std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(ConditionClasses, TestMultiMessageAvailableConditionPerReceiver) {
  // Test supplying sampling_mode argument as a std::string
  Fragment F;
  const std::string name{"multi-message-available"};
  ArgList arglist{Arg{"sampling_mode", std::string("PerReceiver")},
                  Arg{"min_sizes", std::vector<size_t>({1, 2, 1})}};
  auto condition = F.make_condition<MultiMessageAvailableCondition>(name, arglist);
  EXPECT_EQ(condition->name(), name);
  EXPECT_EQ(typeid(condition), typeid(std::make_shared<MultiMessageAvailableCondition>(arglist)));
  EXPECT_EQ(std::string(condition->gxf_typename()),
            "nvidia::gxf::MultiMessageAvailableSchedulingTerm"s);
  EXPECT_TRUE(condition->description().find("name: " + name) != std::string::npos);

  // verify no error logged about failure to convert sampling_mode to YAML::Node was logged
  testing::internal::CaptureStderr();

  condition->initialize();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Unable to convert argument type") == std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(ConditionClasses, TestMultiMessageAvailableConditionPerReceiverYAML) {
  // Test supplying sampling_mode argument as a YAML::Node
  Fragment F;
  const std::string name{"multi-message-available"};
  ArgList arglist{Arg{"sampling_mode", YAML::Node(std::string("PerReceiver"))},
                  Arg{"min_sizes", std::vector<size_t>({1, 2, 1})}};
  auto condition = F.make_condition<MultiMessageAvailableCondition>(name, arglist);
  EXPECT_EQ(condition->name(), name);
  EXPECT_EQ(typeid(condition), typeid(std::make_shared<MultiMessageAvailableCondition>(arglist)));
  EXPECT_EQ(std::string(condition->gxf_typename()),
            "nvidia::gxf::MultiMessageAvailableSchedulingTerm"s);
  EXPECT_TRUE(condition->description().find("name: " + name) != std::string::npos);

  // verify no error logged about failure to convert sampling_mode to YAML::Node was logged
  testing::internal::CaptureStderr();

  condition->initialize();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Unable to convert argument type") == std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(ConditionClasses, TestMultiMessageAvailableConditionDefaultConstructor) {
  Fragment F;
  auto condition = F.make_condition<MultiMessageAvailableCondition>();
}

TEST(ConditionClasses, TestMultiMessageAvailableTimeoutConditionSumOfAll) {
  Fragment F;
  const std::string name{"multi-message-available-timeout"};
  ArgList arglist{Arg{"execution_frequency", "1000000"},
                  Arg{"sampling_mode", std::string("SumOfAll")},
                  Arg{"min_sum", static_cast<size_t>(4)}};
  auto condition = F.make_condition<MultiMessageAvailableTimeoutCondition>(name, arglist);
  EXPECT_EQ(condition->name(), name);
  EXPECT_EQ(typeid(condition),
            typeid(std::make_shared<MultiMessageAvailableTimeoutCondition>(arglist)));
  EXPECT_EQ(std::string(condition->gxf_typename()),
            "nvidia::gxf::MessageAvailableFrequencyThrottler"s);
  EXPECT_TRUE(condition->description().find("name: " + name) != std::string::npos);

  // verify no error logged about failure to convert sampling_mode to YAML::Node was logged
  testing::internal::CaptureStderr();

  condition->initialize();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Unable to convert argument type") == std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(ConditionClasses, TestMultiMessageAvailableTimeoutConditionPerReceiver) {
  // Test supplying sampling_mode argument as an enum
  Fragment F;
  const std::string name{"multi-message-available-timeout"};
  ArgList arglist{
      Arg{"execution_frequency", "10ms"},
      Arg{"sampling_mode", MultiMessageAvailableTimeoutCondition::SamplingMode::kPerReceiver},
      Arg{"min_sizes", std::vector<size_t>({1, 2, 1})}};
  auto condition = F.make_condition<MultiMessageAvailableTimeoutCondition>(name, arglist);
  EXPECT_EQ(condition->name(), name);
  EXPECT_EQ(typeid(condition),
            typeid(std::make_shared<MultiMessageAvailableTimeoutCondition>(arglist)));
  EXPECT_EQ(std::string(condition->gxf_typename()),
            "nvidia::gxf::MessageAvailableFrequencyThrottler"s);
  EXPECT_TRUE(condition->description().find("name: " + name) != std::string::npos);

  // verify no error logged about failure to convert sampling_mode to YAML::Node was logged
  testing::internal::CaptureStderr();

  condition->initialize();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Unable to convert argument type") == std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(ConditionClasses, TestMultiMessageAvailableTimeoutConditionPerReceiverString) {
  // Test supplying sampling_mode argument as a string
  Fragment F;
  const std::string name{"multi-message-available-timeout"};
  ArgList arglist{Arg{"execution_frequency", "10ms"},
                  Arg{"sampling_mode", std::string("PerReceiver")},
                  Arg{"min_sizes", std::vector<size_t>({1, 2, 1})}};
  auto condition = F.make_condition<MultiMessageAvailableTimeoutCondition>(name, arglist);
  EXPECT_EQ(condition->name(), name);
  EXPECT_EQ(typeid(condition),
            typeid(std::make_shared<MultiMessageAvailableTimeoutCondition>(arglist)));
  EXPECT_EQ(std::string(condition->gxf_typename()),
            "nvidia::gxf::MessageAvailableFrequencyThrottler"s);
  EXPECT_TRUE(condition->description().find("name: " + name) != std::string::npos);

  // verify no error logged about failure to convert sampling_mode to YAML::Node was logged
  testing::internal::CaptureStderr();

  condition->initialize();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Unable to convert argument type") == std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(ConditionClasses, TestPeriodicCondition) {
  Fragment F;
  const std::string name{"periodic-condition"};
  auto condition = F.make_condition<PeriodicCondition>(
      name,
      Arg{"recess_period", std::string("1000000")},
      Arg{"policy", PeriodicConditionPolicy::kMinTimeBetweenTicks});
  EXPECT_EQ(condition->name(), name);
  EXPECT_EQ(typeid(condition), typeid(std::make_shared<PeriodicCondition>(1000000)));
  EXPECT_EQ(std::string(condition->gxf_typename()), "nvidia::gxf::PeriodicSchedulingTerm"s);
  EXPECT_TRUE(condition->description().find("name: " + name) != std::string::npos);
  // have to initialize before checking policy (in normal usage GXFExecutor/Operator would do this)
  condition->initialize();
  EXPECT_EQ(condition->policy(), PeriodicConditionPolicy::kMinTimeBetweenTicks);
}

TEST(ConditionClasses, TestPeriodicConditionPolicyAsString) {
  Fragment F;
  const std::string name{"periodic-condition"};
  auto condition = F.make_condition<PeriodicCondition>(
      name, Arg{"recess_period", std::string("1000000")}, Arg{"policy", "NoCatchUpMissedTicks"});
  EXPECT_EQ(condition->name(), name);
  EXPECT_EQ(typeid(condition), typeid(std::make_shared<PeriodicCondition>(1000000)));
  EXPECT_EQ(std::string(condition->gxf_typename()), "nvidia::gxf::PeriodicSchedulingTerm"s);
  EXPECT_TRUE(condition->description().find("name: " + name) != std::string::npos);
  // have to initialize before checking policy (in normal usage GXFExecutor/Operator would do this)
  condition->initialize();
  EXPECT_EQ(condition->policy(), PeriodicConditionPolicy::kNoCatchUpMissedTicks);
}

TEST(ConditionClasses, TestPeriodicConditionDefaultConstructor) {
  Fragment F;
  auto condition = F.make_condition<PeriodicCondition>();
}

TEST(ConditionClasses, TestPeriodicConditionConstructors) {
  using namespace std::chrono_literals;
  Fragment F;
  auto condition = F.make_condition<PeriodicCondition>("1s");
  EXPECT_EQ(
      typeid(condition),
      typeid(std::make_shared<PeriodicCondition>(Arg{"recess_period", std::string("1000000000")})));

  condition = F.make_condition<PeriodicCondition>(1000000000);
  EXPECT_EQ(
      typeid(condition),
      typeid(std::make_shared<PeriodicCondition>(Arg{"recess_period", std::string("1000000000")})));

  condition = F.make_condition<PeriodicCondition>(std::chrono::duration<int, std::kilo>(1));
  EXPECT_EQ(typeid(condition),
            typeid(std::make_shared<PeriodicCondition>(
                Arg{"recess_period", std::string("1000000000000")})));

  condition = F.make_condition<PeriodicCondition>(std::chrono::duration<double, std::milli>(1));
  EXPECT_EQ(
      typeid(condition),
      typeid(std::make_shared<PeriodicCondition>(Arg{"recess_period", std::string("1000000")})));

  condition = F.make_condition<PeriodicCondition>(std::chrono::duration<double, std::micro>(1));
  EXPECT_EQ(typeid(condition),
            typeid(std::make_shared<PeriodicCondition>(Arg{"recess_period", std::string("1000")})));

  condition = F.make_condition<PeriodicCondition>(std::chrono::duration<double, std::nano>(1000));
  EXPECT_EQ(typeid(condition),
            typeid(std::make_shared<PeriodicCondition>(Arg{"recess_period", std::string("1000")})));

  condition =
      F.make_condition<PeriodicCondition>(std::chrono::duration<double, std::ratio<1, 50>>(1));
  EXPECT_EQ(
      typeid(condition),
      typeid(std::make_shared<PeriodicCondition>(Arg{"recess_period", std::string("20000000")})));
}

TEST(ConditionClasses, TestPeriodicConditionMethods) {
  using namespace std::chrono_literals;
  Fragment F;
  const std::string name{"periodic-condition"};
  auto condition =
      F.make_condition<PeriodicCondition>(name, Arg{"recess_period", std::string("1000000")});

  condition->recess_period(1000000);
  EXPECT_EQ(condition->recess_period_ns(), 1000000);

  condition->recess_period(std::chrono::duration<int, std::kilo>(1));
  EXPECT_EQ(condition->recess_period_ns(), 1000000000000LL);

  condition->recess_period(1min);
  EXPECT_EQ(condition->recess_period_ns(), 60000000000LL);

  condition->recess_period(1s);
  EXPECT_EQ(condition->recess_period_ns(), 1000000000);

  condition->recess_period(1ms);
  EXPECT_EQ(condition->recess_period_ns(), 1000000);

  condition->recess_period(1us);
  EXPECT_EQ(condition->recess_period_ns(), 1000);

  condition->recess_period(1ns);
  EXPECT_EQ(condition->recess_period_ns(), 1);

  condition->recess_period(std::chrono::duration<double, std::ratio<1, 50>>(1));
  EXPECT_EQ(condition->recess_period_ns(), 20000000);

  // can access methods of GXFComponent
  EXPECT_EQ(std::string(condition->gxf_typename()), "nvidia::gxf::PeriodicSchedulingTerm"s);
}

TEST(ConditionClasses, TestPeriodicConditionGXFComponentMethods) {
  Fragment F;
  const std::string name{"periodic-condition"};
  auto condition =
      F.make_condition<PeriodicCondition>(name, Arg{"recess_period", std::string("1000000")});

  // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
  auto gxf_typename = condition->gxf_typename();
  auto context = condition->gxf_context();
  auto cid = condition->gxf_cid();
  auto eid = condition->gxf_eid();
  // NOLINTEND(clang-analyzer-deadcode.DeadStores)
}

TEST_F(ConditionClassesWithGXFContext, TestPeriodicConditionInitializeWithoutSpec) {
  PeriodicCondition periodic{1000000};
  periodic.fragment(&F);
  // TODO(unknown): avoid segfault if initialize is called before the fragment is assigned

  // test that an error is logged if initialize is called before a spec as assigned
  testing::internal::CaptureStderr();
  periodic.initialize();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") != std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("No component spec") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST_F(ConditionClassesWithGXFContext, TestPeriodicConditionInitializeWithArg) {
  using namespace std::chrono_literals;

  auto context = F.executor().context();
  // GXF's PeriodicSchedulingTerm accepts the following recess period units:
  auto pairs = std::vector<std::pair<std::string, int64_t>>{
      {"1000000", 1000000}, {"20hz", 50000000}, {"1s", 1000000000}, {"1ms", 1000000}};

  const GxfEntityCreateInfo entity_create_info = {"dummy_entity", GXF_ENTITY_CREATE_PROGRAM_BIT};
  gxf_uid_t eid = 0;
  gxf_result_t code;
  code = GxfCreateEntity(context, &entity_create_info, &eid);
  ASSERT_EQ(code, GXF_SUCCESS);

  std::vector<std::shared_ptr<PeriodicCondition>> conditions;
  for (auto& pair : pairs) {
    const std::string condition_name = fmt::format("periodic-condition_{}", pair.first);

    auto condition =
        F.make_condition<PeriodicCondition>(condition_name, Arg{"recess_period", pair.first});
    condition->fragment(&F);
    condition->gxf_eid(eid);
    condition->initialize();

    conditions.push_back(condition);
  }

  // Activate the graph to initialize the conditions
  code = GxfGraphActivate(context);
  EXPECT_EQ(code, GXF_SUCCESS);

  for (int i = 0; i < pairs.size(); ++i) {
    auto& pair = pairs[i];
    auto& condition = conditions[i];
    EXPECT_EQ(condition->recess_period_ns(), pair.second);
  }

  // Deactivate the graph
  code = GxfGraphDeactivate(context);
  EXPECT_EQ(code, GXF_SUCCESS);
}

TEST_F(ConditionClassesWithGXFContext, TestPeriodicConditionInitializeWithUnrecognizedArg) {
  auto condition = F.make_condition<PeriodicCondition>(Arg{"recess_period", std::string("1000000")},
                                                       Arg("undefined_arg", 5.0));

  // test that an warning is logged if an unknown argument is provided
  testing::internal::CaptureStderr();
  condition->initialize();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("warning") != std::string::npos) << "=== LOG ===\n"
                                                               << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("'undefined_arg' not found in spec_.params") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(ConditionClasses, TestCudaBufferAvailableCondition) {
  Fragment F;
  const std::string name{"cuda-buffer-available-condition"};
  auto condition = F.make_condition<CudaBufferAvailableCondition>(name);
  EXPECT_EQ(condition->name(), name);
  EXPECT_EQ(typeid(condition), typeid(std::make_shared<CudaBufferAvailableCondition>()));
  EXPECT_EQ(std::string(condition->gxf_typename()),
            "nvidia::gxf::CudaBufferAvailableSchedulingTerm"s);
  EXPECT_TRUE(condition->description().find("name: " + name) != std::string::npos);
}

TEST(ConditionClasses, TestCudaBufferAvailableConditionDefaultConstructor) {
  Fragment F;
  auto condition = F.make_condition<CudaBufferAvailableCondition>();
}

TEST(ConditionClasses, TestCudaStreamCondition) {
  Fragment F;
  const std::string name{"cuda-stream-condition"};
  auto condition = F.make_condition<CudaStreamCondition>(name);
  EXPECT_EQ(condition->name(), name);
  EXPECT_EQ(typeid(condition), typeid(std::make_shared<CudaStreamCondition>()));
  EXPECT_EQ(std::string(condition->gxf_typename()), "nvidia::gxf::CudaStreamSchedulingTerm"s);
  EXPECT_TRUE(condition->description().find("name: " + name) != std::string::npos);
}

TEST(ConditionClasses, TestCudaStreamConditionDefaultConstructor) {
  Fragment F;
  auto condition = F.make_condition<CudaStreamCondition>();
}

TEST(ConditionClasses, TestCudaEventCondition) {
  Fragment F;
  const std::string name{"cuda-event-condition"};
  const std::string event_name{"cuda-event"};
  auto condition = F.make_condition<CudaEventCondition>(name, Arg{"event_name", event_name});
  EXPECT_EQ(condition->name(), name);
  EXPECT_EQ(typeid(condition), typeid(std::make_shared<CudaEventCondition>()));
  EXPECT_EQ(std::string(condition->gxf_typename()), "nvidia::gxf::CudaEventSchedulingTerm"s);
  EXPECT_TRUE(condition->description().find("name: " + name) != std::string::npos);
}

TEST(ConditionClasses, TestCudaEventConditionDefaultConstructor) {
  Fragment F;
  auto condition = F.make_condition<CudaEventCondition>();
}

}  // namespace holoscan
