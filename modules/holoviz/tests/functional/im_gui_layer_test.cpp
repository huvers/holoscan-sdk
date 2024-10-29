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

#include <gtest/gtest.h>
#include <imgui.h>

#include <vector>

#include <holoviz/holoviz.hpp>
#include <util/unique_value.hpp>
#include "test_fixture.hpp"

namespace viz = holoscan::viz;

// Fixture that initializes Holoviz
class ImGuiLayer : public TestHeadless {
 protected:
  ImGuiLayer() : TestHeadless(256, 256) {}
};

TEST_F(ImGuiLayer, Window) {
  for (uint32_t i = 0; i < 2; ++i) {
    EXPECT_NO_THROW(viz::Begin());
    EXPECT_NO_THROW(viz::BeginImGuiLayer());

    ImGui::Begin("Window");
    ImGui::Text("Some Text");
    ImGui::End();

    EXPECT_NO_THROW(viz::EndLayer());
    EXPECT_NO_THROW(viz::End());
  }

  CompareColorResultCRC32({
      0xf9db0778,  // RTX 6000, RTX A5000
      0x0a26e40d   // RTX A6000
  });
}

TEST_F(ImGuiLayer, Errors) {
  std::vector<float> data{0.5F, 0.5F};

  EXPECT_NO_THROW(viz::Begin());

  // it's an error to call BeginImGuiLayer if no valid ImGui context is set
  ImGuiContext* prev_context = ImGui::GetCurrentContext();
  ImGui::SetCurrentContext(nullptr);
  EXPECT_THROW(viz::BeginImGuiLayer(), std::runtime_error);
  ImGui::SetCurrentContext(prev_context);

  // it's an error to call BeginImGuiLayer again without calling EndLayer
  EXPECT_NO_THROW(viz::BeginImGuiLayer());
  EXPECT_THROW(viz::BeginImGuiLayer(), std::runtime_error);
  EXPECT_NO_THROW(viz::EndLayer());

  // multiple ImGui layers per frame are not supported
  EXPECT_THROW(viz::BeginImGuiLayer(), std::runtime_error);

  EXPECT_NO_THROW(viz::End());
}
