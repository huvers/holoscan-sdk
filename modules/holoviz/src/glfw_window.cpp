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

#include "glfw_window.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <backends/imgui_impl_glfw.h>
#include <unistd.h>

#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include <holoscan/logger/logger.hpp>
#include <nvh/cameramanipulator.hpp>
#include <nvh/timesampler.hpp>
#include <nvvk/error_vk.hpp>

namespace holoscan::viz {

static void glfw_error_callback(int error, const char* description) {
  std::cerr << "Glfw Error " << error << ": " << description << std::endl;
}

struct GLFWWindow::Impl {
 public:
  explicit Impl(InitFlags init_flags) :
    init_flags_(init_flags) {
    std::lock_guard<std::mutex> guard(mutex_);

    if (glfw_init_count_ == 0) {
      glfwSetErrorCallback(glfw_error_callback);

      if (glfwInit() == GLFW_FALSE) { throw std::runtime_error("Failed to initialize glfw"); }
    }
    ++glfw_init_count_;

    if (!glfwVulkanSupported()) { throw std::runtime_error("Vulkan is not supported"); }
  }
  Impl() = delete;

  ~Impl() {
    if (intern_window_ && window_) {
      if (init_flags_ & InitFlags::FULLSCREEN) {
        // GLFW is not switching back to the original mode when just destroying the window,
        // have to set the window monitor explicitly to NULL before destroy to switch back
        // to the original mode.
        glfwSetWindowMonitor(window_, NULL, 0, 0, width_, height_, GLFW_DONT_CARE);
      }
      glfwDestroyWindow(window_);
    }

    std::lock_guard<std::mutex> guard(mutex_);
    --glfw_init_count_;
    if (glfw_init_count_ == 0) { glfwTerminate(); }
  }

  static void frame_buffer_size_cb(GLFWwindow* window, int width, int height);
  static void key_cb(GLFWwindow* window, int key, int scancode, int action, int mods);
  static void cursor_pos_cb(GLFWwindow* window, double x, double y);
  static void mouse_button_cb(GLFWwindow* window, int button, int action, int mods);
  static void scroll_cb(GLFWwindow* window, double x, double y);

  const InitFlags init_flags_;

  std::mutex mutex_;                 ///< mutex to protect glfw init counter
  static uint32_t glfw_init_count_;  ///< glfw init counter

  GLFWwindow* window_ = nullptr;
  bool intern_window_ = false;

  std::function<void(int width, int height)> frame_buffer_size_cb_;
  GLFWframebuffersizefun prev_frame_buffer_size_cb_ = nullptr;

  GLFWkeyfun prev_key_cb_ = nullptr;
  GLFWcursorposfun prev_cursor_pos_cb_ = nullptr;
  GLFWmousebuttonfun prev_mouse_button_cb_ = nullptr;
  GLFWscrollfun prev_scroll_cb_ = nullptr;

  nvh::CameraManipulator::Inputs inputs_;  ///< Mouse button pressed
  nvh::Stopwatch timer_;  ///< measure time from frame to frame to base camera movement on

  uint32_t width_ = 0;
  uint32_t height_ = 0;
};

uint32_t GLFWWindow::Impl::glfw_init_count_ = 0;

GLFWWindow::GLFWWindow(GLFWwindow* window) : impl_(new Impl(InitFlags::NONE)) {
  impl_->window_ = window;

  // set the user pointer to the implementation class to be used in callbacks, fail if the provided
  // window already has the user pointer set.
  if (glfwGetWindowUserPointer(impl_->window_) != nullptr) {
    throw std::runtime_error("GLFW window user pointer already set");
  }
  glfwSetWindowUserPointer(impl_->window_, impl_.get());

  // set framebuffer size with initial window size
  int width, height;
  glfwGetWindowSize(window, &width, &height);
  impl_->frame_buffer_size_cb(window, width, height);
}

GLFWWindow::GLFWWindow(uint32_t width, uint32_t height, const char* title, InitFlags flags,
                       const char* display_name)
    : impl_(new Impl(flags)) {
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  GLFWmonitor* monitor = nullptr;
  if (flags & InitFlags::FULLSCREEN) {
    if (display_name) {
      int monitor_count = 0;
      GLFWmonitor** monitors = glfwGetMonitors(&monitor_count);
      for (int index = 0; index < monitor_count; ++index) {
        const char* monitor_name = glfwGetMonitorName(monitors[index]);
        if (std::strcmp(monitor_name, display_name) == 0) {
          monitor = monitors[index];
          break;
        }
      }
      if (!monitor) {
        HOLOSCAN_LOG_WARN("Display \"{}\" not found, using the primary display instead",
                          display_name);
        HOLOSCAN_LOG_INFO("____________________");
        HOLOSCAN_LOG_INFO("Available displays :");
        for (int index = 0; index < monitor_count; ++index) {
          HOLOSCAN_LOG_INFO("{}", glfwGetMonitorName(monitors[index]));
        }
      }
    }
    if (!monitor) { monitor = glfwGetPrimaryMonitor(); }
  }

  impl_->window_ = glfwCreateWindow(width, height, title, monitor, NULL);
  if (!impl_->window_) { throw std::runtime_error("Failed to create glfw window"); }

  impl_->intern_window_ = true;

  // set the user pointer to the implementation class to be used in callbacks
  glfwSetWindowUserPointer(impl_->window_, impl_.get());

  // set framebuffer size with initial window size
  impl_->frame_buffer_size_cb(impl_->window_, width, height);
}

GLFWWindow::~GLFWWindow() {}

void GLFWWindow::init_im_gui() {
  ImGui_ImplGlfw_InitForVulkan(impl_->window_, true);
}

void GLFWWindow::Impl::frame_buffer_size_cb(GLFWwindow* window, int width, int height) {
  GLFWWindow::Impl* const impl = static_cast<GLFWWindow::Impl*>(glfwGetWindowUserPointer(window));

  if (impl->prev_frame_buffer_size_cb_) { impl->prev_frame_buffer_size_cb_(window, width, height); }

  if (impl->frame_buffer_size_cb_) { impl->frame_buffer_size_cb_(width, height); }

  impl->width_ = width;
  impl->height_ = height;
  CameraManip.setWindowSize(width, height);
}

void GLFWWindow::Impl::key_cb(GLFWwindow* window, int key, int scancode, int action, int mods) {
  GLFWWindow::Impl* const impl = static_cast<GLFWWindow::Impl*>(glfwGetWindowUserPointer(window));

  if (impl->prev_key_cb_) { impl->prev_key_cb_(window, key, scancode, action, mods); }

  const bool pressed = action != GLFW_RELEASE;

  if (pressed && (key == GLFW_KEY_ESCAPE)) { glfwSetWindowShouldClose(window, 1); }

  // Keeping track of the modifiers
  impl->inputs_.ctrl =
      pressed & ((key == GLFW_KEY_LEFT_CONTROL) || (key == GLFW_KEY_RIGHT_CONTROL));
  impl->inputs_.shift = pressed & ((key == GLFW_KEY_LEFT_SHIFT) || (key == GLFW_KEY_RIGHT_SHIFT));
  impl->inputs_.alt = pressed & ((key == GLFW_KEY_LEFT_ALT) || (key == GLFW_KEY_RIGHT_ALT));
}

void GLFWWindow::Impl::mouse_button_cb(GLFWwindow* window, int button, int action, int mods) {
  GLFWWindow::Impl* const impl = static_cast<GLFWWindow::Impl*>(glfwGetWindowUserPointer(window));

  if (impl->prev_mouse_button_cb_) { impl->prev_mouse_button_cb_(window, button, action, mods); }

  double x, y;
  glfwGetCursorPos(impl->window_, &x, &y);
  CameraManip.setMousePosition(static_cast<int>(x), static_cast<int>(y));

  impl->inputs_.lmb = (button == GLFW_MOUSE_BUTTON_LEFT) && (action == GLFW_PRESS);
  impl->inputs_.mmb = (button == GLFW_MOUSE_BUTTON_MIDDLE) && (action == GLFW_PRESS);
  impl->inputs_.rmb = (button == GLFW_MOUSE_BUTTON_RIGHT) && (action == GLFW_PRESS);
}

void GLFWWindow::Impl::cursor_pos_cb(GLFWwindow* window, double x, double y) {
  GLFWWindow::Impl* const impl = static_cast<GLFWWindow::Impl*>(glfwGetWindowUserPointer(window));

  if (impl->prev_cursor_pos_cb_) { impl->prev_cursor_pos_cb_(window, x, y); }

  // Allow camera movement only when not editing
  if ((ImGui::GetCurrentContext() != nullptr) && ImGui::GetIO().WantCaptureMouse) { return; }

  if (impl->inputs_.lmb || impl->inputs_.rmb || impl->inputs_.mmb) {
    CameraManip.mouseMove(static_cast<int>(x), static_cast<int>(y), impl->inputs_);
  }
}

void GLFWWindow::Impl::scroll_cb(GLFWwindow* window, double x, double y) {
  GLFWWindow::Impl* const impl = static_cast<GLFWWindow::Impl*>(glfwGetWindowUserPointer(window));

  if (impl->prev_scroll_cb_) { impl->prev_scroll_cb_(window, x, y); }

  // Allow camera movement only when not editing
  if ((ImGui::GetCurrentContext() != nullptr) && ImGui::GetIO().WantCaptureMouse) { return; }

  CameraManip.wheel(y > 0.0 ? 1 : -1, impl->inputs_);
}

void GLFWWindow::setup_callbacks(std::function<void(int width, int height)> frame_buffer_size_cb) {
  impl_->frame_buffer_size_cb_ = std::move(frame_buffer_size_cb);

  impl_->prev_frame_buffer_size_cb_ =
      glfwSetFramebufferSizeCallback(impl_->window_, &GLFWWindow::Impl::frame_buffer_size_cb);
  impl_->prev_mouse_button_cb_ =
      glfwSetMouseButtonCallback(impl_->window_, &GLFWWindow::Impl::mouse_button_cb);
  impl_->prev_scroll_cb_ = glfwSetScrollCallback(impl_->window_, &GLFWWindow::Impl::scroll_cb);
  impl_->prev_cursor_pos_cb_ =
      glfwSetCursorPosCallback(impl_->window_, &GLFWWindow::Impl::cursor_pos_cb);
  impl_->prev_key_cb_ = glfwSetKeyCallback(impl_->window_, &GLFWWindow::Impl::key_cb);
}

void GLFWWindow::restore_callbacks() {
  glfwSetFramebufferSizeCallback(impl_->window_, impl_->prev_frame_buffer_size_cb_);
  glfwSetMouseButtonCallback(impl_->window_, impl_->prev_mouse_button_cb_);
  glfwSetScrollCallback(impl_->window_, impl_->prev_scroll_cb_);
  glfwSetCursorPosCallback(impl_->window_, impl_->prev_cursor_pos_cb_);
  glfwSetKeyCallback(impl_->window_, impl_->prev_key_cb_);

  impl_->frame_buffer_size_cb_ = nullptr;
  impl_->prev_key_cb_ = nullptr;
  impl_->prev_cursor_pos_cb_ = nullptr;
  impl_->prev_mouse_button_cb_ = nullptr;
  impl_->prev_scroll_cb_ = nullptr;
}

const char** GLFWWindow::get_required_instance_extensions(uint32_t* count) {
  return glfwGetRequiredInstanceExtensions(count);
}

const char** GLFWWindow::get_required_device_extensions(uint32_t* count) {
  static char const* extensions[]{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  *count = sizeof(extensions) / sizeof(extensions[0]);
  return extensions;
}

uint32_t GLFWWindow::select_device(vk::Instance instance,
                                   const std::vector<vk::PhysicalDevice>& physical_devices) {
  // select the first device which has presentation support
  for (uint32_t index = 0; index < physical_devices.size(); ++index) {
    if (glfwGetPhysicalDevicePresentationSupport(instance, physical_devices[index], 0) ==
        GLFW_TRUE) {
      return index;
    }
  }
  throw std::runtime_error("No device with presentation support found");
}

void GLFWWindow::get_framebuffer_size(uint32_t* width, uint32_t* height) {
  *width = impl_->width_;
  *height = impl_->height_;
}

vk::SurfaceKHR GLFWWindow::create_surface(vk::PhysicalDevice physical_device,
                                          vk::Instance instance) {
  VkSurfaceKHR surface;
  const vk::Result result =
      vk::Result(glfwCreateWindowSurface(instance, impl_->window_, nullptr, &surface));
  if (result != vk::Result::eSuccess) {
    vk::throwResultException(result, "Failed to create glfw window surface");
  }
  return surface;
}

bool GLFWWindow::should_close() {
  return (glfwWindowShouldClose(impl_->window_) != 0);
}

bool GLFWWindow::is_minimized() {
  bool minimized(impl_->width_ == 0 || impl_->height_ == 0);
  if (minimized) { usleep(50); }
  return minimized;
}

void GLFWWindow::im_gui_new_frame() {
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
}

void GLFWWindow::begin() {
  glfwPollEvents();
}

void GLFWWindow::end() {
  // call the base class
  Window::end();
}

float GLFWWindow::get_aspect_ratio() {
  if (impl_->height_) {
    return float(impl_->width_) / float(impl_->height_);
  } else {
    return 1.f;
  }
}

}  // namespace holoscan::viz
