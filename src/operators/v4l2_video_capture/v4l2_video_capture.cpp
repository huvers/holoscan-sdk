/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/operators/v4l2_video_capture/v4l2_video_capture.hpp"

#include <fcntl.h>
#include <jpeglib.h>
#include <libv4l2.h>
#include <libv4lconvert.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <numeric>
#include <string>
#include <utility>

#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"

#define CLEAR(x) memset(&(x), 0, sizeof(x))

bool pixel_format_supported(int fd, unsigned int pixel_format_fourcc) {
  struct v4l2_fmtdesc fmtdesc;
  CLEAR(fmtdesc);
  fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  bool supported_format = false;
  while (ioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc) == 0) {
    if (fmtdesc.pixelformat == pixel_format_fourcc) {
      supported_format = true;
      break;
    }
    fmtdesc.index++;
  }
  return supported_format;
}

namespace holoscan::ops {

void V4L2VideoCaptureOp::setup(OperatorSpec& spec) {
  auto& signal = spec.output<gxf::Entity>("signal");

  spec.param(signal_, "signal", "Output", "Output channel", &signal);

  static constexpr char kDefaultDevice[] = "/dev/video0";
  static constexpr char kDefaultPixelFormat[] = "auto";
  static constexpr uint32_t kDefaultWidth = 0;
  static constexpr uint32_t kDefaultHeight = 0;
  static constexpr uint32_t kDefaultNumBuffers = 4;

  spec.param(allocator_, "allocator", "Allocator", "Output Allocator");

  spec.param(
      device_, "device", "VideoDevice", "Path to the V4L2 device", std::string(kDefaultDevice));
  spec.param(width_, "width", "Width", "Width of the V4L2 image", kDefaultWidth);
  spec.param(height_, "height", "Height", "Height of the V4L2 image", kDefaultHeight);
  spec.param(num_buffers_,
             "numBuffers",
             "NumBuffers",
             "Number of V4L2 buffers to use",
             kDefaultNumBuffers);
  spec.param(pixel_format_,
             "pixel_format",
             "Pixel Format",
             "Pixel format of capture stream (little endian four character code (fourcc))",
             std::string(kDefaultPixelFormat));
  spec.param(exposure_time_,
             "exposure_time",
             "Exposure Time",
             "Exposure time of the camera sensor in multiples of 100 μs (e.g. setting "
             "exposure_time to 100 is 10 ms). See V4L2_CID_EXPOSURE_ABSOLUTE.",
             ParameterFlag::kOptional);
  spec.param(gain_,
             "gain",
             "Gain",
             "Gain of the camera sensor. See V4L2_CID_GAIN.",
             ParameterFlag::kOptional);
}

void V4L2VideoCaptureOp::initialize() {
  Operator::initialize();
}

void V4L2VideoCaptureOp::start() {
  v4l2_initialize();
  v4l2_set_mode();
  v4l2_check_formats();
  v4l2_set_formats();
  v4l2_set_camera_settings();
  v4l2_requestbuffers();
  v4l2_start();
}

void V4L2VideoCaptureOp::compute(InputContext& op_input, OutputContext& op_output,
                                 ExecutionContext& context) {
  // Avoid warning about unused variable
  (void)op_input;

  // Read buffer.
  struct v4l2_buffer buf;
  CLEAR(buf);
  v4l2_read_buffer(buf);

  // Create video buffer
  auto out_message = nvidia::gxf::Entity::New(context.context());
  if (!out_message) { throw std::runtime_error("Failed to allocate video output; terminating."); }
  auto video_buffer = out_message.value().add<nvidia::gxf::VideoBuffer>();
  if (!video_buffer) { throw std::runtime_error("Failed to allocate video buffer; terminating."); }

  // Get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
  auto allocator =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());
  // Allocate output buffer
  video_buffer.value()->resize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA>(
      width_use_,
      height_use_,
      nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR,
      nvidia::gxf::MemoryStorageType::kHost,
      allocator.value(),
      false);
  if (!video_buffer.value()->pointer()) {
    throw std::runtime_error("Failed to allocate output buffer.");
  }

  // Wrap buffer
  Buffer& read_buf = buffers_[buf.index];
  if (pixel_format_use_ == V4L2_PIX_FMT_YUYV) {
    // Convert YUYV to RGBA output buffer
    YUYVToRGBA(read_buf.ptr, video_buffer.value()->pointer(), width_use_, height_use_);

    // Return (queue) the buffer.
    if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
      throw std::runtime_error(
          fmt::format("Failed to queue buffer {} on {}", buf.index, device_.get().c_str()));
    }
  } else if (pixel_format_use_ == V4L2_PIX_FMT_MJPEG) {
    // Convert MJPG to RGBA output buffer
    MJPEGToRGBA(read_buf.ptr, video_buffer.value()->pointer(), width_use_, height_use_);

    // Return (queue) the buffer.
    if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
      throw std::runtime_error(
          fmt::format("Failed to queue buffer {} on {}", buf.index, device_.get().c_str()));
    }
  } else {
    // Wrap memory into output buffer
    video_buffer.value()->wrapMemory(
        video_buffer.value()->video_frame_info(),
        buf.length,
        nvidia::gxf::MemoryStorageType::kHost,
        read_buf.ptr,
        [buffer = buf, fd = fd_, device = device_](void*) mutable {
          if (ioctl(fd, VIDIOC_QBUF, &buffer) < 0) {
            throw std::runtime_error(
                fmt::format("Failed to queue buffer {} on {}", buffer.index, device.get().c_str()));
          }
          return nvidia::gxf::Success;
        });
  }

  auto result = gxf::Entity(std::move(out_message.value()));
  op_output.emit(result);
}

void V4L2VideoCaptureOp::stop() {
  // stream off
  enum v4l2_buf_type buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (-1 == ioctl(fd_, VIDIOC_STREAMOFF, &buf_type)) {
    throw std::runtime_error("StreamOFF Ioctl Failed");
  }

  // free buffers
  for (uint32_t i = 0; i < num_buffers_.get(); ++i)
    if (-1 == munmap(buffers_[i].ptr, buffers_[i].length)) {
      throw std::runtime_error(fmt::format("munmap Failed for index {}", i));
    }
  free(buffers_);

  // close FD
  if (-1 == v4l2_close(fd_)) { throw std::runtime_error("Close failed"); }

  fd_ = -1;
}

void V4L2VideoCaptureOp::v4l2_initialize() {
  // Initialise V4L2 device
  fd_ = v4l2_open(device_.get().c_str(), O_RDWR);
  if (fd_ < 0) {
    throw std::runtime_error(
        "Failed to open device! Possible permission issue with accessing the device.");
  }

  struct v4l2_capability caps;
  CLEAR(caps);
  ioctl(fd_, VIDIOC_QUERYCAP, &caps);
  if (!(caps.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
    throw std::runtime_error("No V4l2 Video capture node");
  }
  if (!(caps.capabilities & V4L2_CAP_STREAMING)) {
    throw std::runtime_error("Does not support streaming i/o");
  }
}

void V4L2VideoCaptureOp::v4l2_requestbuffers() {
  // Request V4L2 buffers
  struct v4l2_requestbuffers req;
  CLEAR(req);
  req.count = num_buffers_.get();
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;

  if (-1 == ioctl(fd_, VIDIOC_REQBUFS, &req)) {
    if (errno == EINVAL)
      throw std::runtime_error(fmt::format(
          "Video capturing or DMABUF streaming is not supported type {} memory {} count {}",
          req.type,
          req.memory,
          req.count));
    else
      throw std::runtime_error("Request buffers Ioctl failed");
  }

  buffers_ = (Buffer*)calloc(req.count, sizeof(*buffers_));
  if (!buffers_) { throw std::runtime_error("Allocate buffers failed"); }

  for (uint32_t i = 0; i < req.count; ++i) {
    struct v4l2_buffer buf;
    CLEAR(buf);
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;

    if (-1 == ioctl(fd_, VIDIOC_QUERYBUF, &buf)) {
      throw std::runtime_error("VIDIOC_QUERYBUF Ioctl failed");
    }

    buffers_[i].length = buf.length;
    buffers_[i].ptr = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
    if (MAP_FAILED == buffers_[i].ptr) { throw std::runtime_error("MMAP failed"); }
  }
}

void V4L2VideoCaptureOp::v4l2_check_formats() {
  // Check user-given pixel format (also OK if "auto")
  if (pixel_format_.get().length() != 4) {
    throw std::runtime_error(
        fmt::format("User-given pixel format should have V4L2 FourCC string format, got {}",
                    pixel_format_.get()));
  }

  if (pixel_format_.get() != "auto") {
    // Convert user given format to FourCC
    uint32_t pixel_format = v4l2_fourcc(pixel_format_.get()[0],
                                        pixel_format_.get()[1],
                                        pixel_format_.get()[2],
                                        pixel_format_.get()[3]);

    // Check if the device supports the requested pixel format
    bool supported_format = pixel_format_supported(fd_, pixel_format);
    if (supported_format == false) {
      throw std::runtime_error(
          fmt::format("Device does not support '{}' pixel format", pixel_format_.get()));
    }
    // Update format with valid user-given format
    pixel_format_use_ = pixel_format;
  } else if (pixel_format_.get() == "auto") {
    // Currently, AB24, YUYV, and MJPG are supported in auto mode
    uint32_t ab24 = v4l2_fourcc('A', 'B', '2', '4');
    uint32_t yuyv = v4l2_fourcc('Y', 'U', 'Y', 'V');
    uint32_t mjpg = v4l2_fourcc('M', 'J', 'P', 'G');

    if (pixel_format_supported(fd_, ab24)) {
      pixel_format_use_ = ab24;
    } else if (pixel_format_supported(fd_, yuyv)) {
      pixel_format_use_ = yuyv;
    } else if (pixel_format_supported(fd_, mjpg)) {
      pixel_format_use_ = mjpg;
    } else {
      throw std::runtime_error(
          "Automatic setting of pixel format failed: device does not support AB24, YUYV, or MJPG. "
          " If you are sure that the device pixel format is RGBA, please specify the pixel format "
          "in the yaml configuration file.");
    }
  }

  if (width_.get() > 0 || height_.get() > 0) {
    // Check if the device supports the requested width and height
    struct v4l2_frmsizeenum frmsize;
    CLEAR(frmsize);
    frmsize.pixel_format = pixel_format_use_;
    int supported_formats = 0;
    while (ioctl(fd_, VIDIOC_ENUM_FRAMESIZES, &frmsize) == 0) {
      if (frmsize.type != V4L2_FRMSIZE_TYPE_DISCRETE) {
        throw std::runtime_error("Non-discrete frame sizes not supported");
      }
      if (width_.get() == 0 || frmsize.discrete.width == width_.get()) supported_formats += 1;
      if (height_.get() == 0 || frmsize.discrete.height == height_.get()) supported_formats += 1;
      if (supported_formats == 2) break;
      frmsize.index++;
    }
    if (supported_formats != 2) {
      throw std::runtime_error(
          fmt::format("Device does not support '{}x{}'", width_.get(), height_.get()));
    }
    // Update format with valid user-given format
    if (width_.get() > 0) width_use_ = width_.get();
    if (height_.get() > 0) height_use_ = height_.get();
  }
}

void V4L2VideoCaptureOp::v4l2_set_mode() {
  // Set V4L2 device mode
  struct v4l2_format vfmt;
  CLEAR(vfmt);
  vfmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (ioctl(fd_, VIDIOC_G_FMT, &vfmt) == -1) {
    throw std::runtime_error("Get format Ioctl failed, is device set correctly?");
  }
  // Store default formats
  width_use_ = vfmt.fmt.pix_mp.width;
  height_use_ = vfmt.fmt.pix_mp.height;
  pixel_format_use_ = vfmt.fmt.pix.pixelformat;
}

void V4L2VideoCaptureOp::v4l2_set_formats() {
  // Get formats
  struct v4l2_format vfmt;
  CLEAR(vfmt);
  vfmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (ioctl(fd_, VIDIOC_G_FMT, &vfmt) == -1) {
    throw std::runtime_error("Get format Ioctl failed");
  }
  // Modify pixel format and image size
  vfmt.fmt.pix_mp.width = width_use_;
  vfmt.fmt.pix_mp.height = height_use_;
  vfmt.fmt.pix.pixelformat = pixel_format_use_;
  vfmt.fmt.pix_mp.plane_fmt[0].bytesperline = width_use_ * 4;
  // Set formats
  if (ioctl(fd_, VIDIOC_S_FMT, &vfmt) == -1) {
    if (errno == EINVAL) {
      throw std::runtime_error("Requested buffer type not supported in Set FMT");
    } else {
      throw std::runtime_error(fmt::format("Set FMT Ioctl failed with {}", errno));
    }
  }
}

bool V4L2VideoCaptureOp::v4l2_camera_supports_control(int cid, const char* control_name) {
  struct v4l2_queryctrl queryctrl;

  memset(&queryctrl, 0, sizeof(queryctrl));
  queryctrl.id = cid;

  if (ioctl(fd_, VIDIOC_QUERYCTRL, &queryctrl) == -1) {
    // EINVAL indicates that the control is not supported
    if (errno == EINVAL) {
      return false;
    } else {
      throw std::runtime_error(fmt::format("Querying {} failed", control_name));
    }
  }

  return true;
}

// Set the device's control settings if supported
void V4L2VideoCaptureOp::v4l2_set_camera_control(v4l2_control control, const char* control_name,
                                                 bool warn) {
  HOLOSCAN_LOG_DEBUG(fmt::format("Setting {} to {}", control_name, control.value));
  if (v4l2_camera_supports_control(control.id, control_name)) {
    if (ioctl(fd_, VIDIOC_S_CTRL, &control) == -1) {
      HOLOSCAN_LOG_DEBUG(fmt::format("Setting {} to {} failed", control_name, control.value));
      throw std::runtime_error(fmt::format("Setting {} to {} failed", control_name, control.value));
    }
  } else {
    auto msg = fmt::format("Device does not support {}", control_name);
    if (warn) {
      HOLOSCAN_LOG_WARN(msg);
    } else {
      HOLOSCAN_LOG_DEBUG(msg);
    }
  }
}

void V4L2VideoCaptureOp::v4l2_set_camera_settings() {
  struct v4l2_capability caps;
  CLEAR(caps);

  // To check if the dev is v4l2loopback
  if (ioctl(fd_, VIDIOC_QUERYCAP, &caps) == -1) {
    throw std::runtime_error("Querying video capabilities failed");
  }
  std::string busInfo = reinterpret_cast<const char*>(caps.bus_info);
  if (busInfo.find("v4l2loopback") != std::string::npos) {
    // Return before setting the camera parameters as loopback option
    // does not have camera settings to run.
    HOLOSCAN_LOG_DEBUG("Found a v4l2loopback device");
    return;
  }

  struct v4l2_control control;
  CLEAR(control);

  // Set Exposure
  if (exposure_time_.try_get().has_value()) {
    // Manual exposure: try EXPOSURE_SHUTTER_PRIORITY first (manual exposure, auto iris)
    control.id = V4L2_CID_EXPOSURE_AUTO;
    control.value = V4L2_EXPOSURE_SHUTTER_PRIORITY;
    try {
      v4l2_set_camera_control(control, "V4L2_CID_EXPOSURE_AUTO", false);
    } catch (std::exception& e) {
      // If fails, try setting to full manual mode
      control.value = V4L2_EXPOSURE_MANUAL;
      v4l2_set_camera_control(control, "V4L2_CID_EXPOSURE_AUTO", true);
    }
    // Then set the value
    CLEAR(control);
    control.id = V4L2_CID_EXPOSURE_ABSOLUTE;
    control.value = exposure_time_;
    v4l2_set_camera_control(control, "V4L2_CID_EXPOSURE_ABSOLUTE", true);
  } else {
    // Auto exposure: try fully auto first (auto exposure, auto iris)
    control.id = V4L2_CID_EXPOSURE_AUTO;
    control.value = V4L2_EXPOSURE_AUTO;
    try {
      v4l2_set_camera_control(control, "V4L2_CID_EXPOSURE_AUTO", false);
    } catch (std::exception& e) {
      // If fails, try setting to APERTURE_PRIORITY (auto exposure, manual iris)
      control.value = V4L2_EXPOSURE_APERTURE_PRIORITY;
      v4l2_set_camera_control(control, "V4L2_CID_EXPOSURE_AUTO", false);
    }
  }

  // Set Gain
  if (gain_.try_get().has_value()) {
    // Manual: turn auto gain off
    control.id = V4L2_CID_AUTOGAIN;
    control.value = 0;
    v4l2_set_camera_control(control, "V4L2_CID_AUTOGAIN", false);

    // Then set value
    CLEAR(control);
    control.id = V4L2_CID_GAIN;
    control.value = gain_;
    v4l2_set_camera_control(control, "V4L2_CID_GAIN", true);
  } else {
    // Auto gain
    control.id = V4L2_CID_AUTOGAIN;
    control.value = 1;
    v4l2_set_camera_control(control, "V4L2_CID_AUTOGAIN", false);
  }
}

void V4L2VideoCaptureOp::v4l2_start() {
  // Start streaming on V4L2 device
  // queue capture plane into device
  for (uint32_t i = 0; i < num_buffers_.get(); i++) {
    struct v4l2_buffer buf;
    CLEAR(buf);

    buf.index = i;
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (-1 == ioctl(fd_, VIDIOC_QBUF, &buf)) {
      throw std::runtime_error("Failed to queue buf, Ioctl failed");
    }
  }

  enum v4l2_buf_type buf_type;
  buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (-1 == ioctl(fd_, VIDIOC_STREAMON, &buf_type)) {
    throw std::runtime_error("StreamOn Ioctl failed");
  }
}

void V4L2VideoCaptureOp::v4l2_read_buffer(v4l2_buffer& buf) {
  fd_set fds;
  FD_ZERO(&fds);
  FD_SET(fd_, &fds);

  struct timeval tv;
  tv.tv_sec = 15;
  tv.tv_usec = 0;

  int r;
  r = select(fd_ + 1, &fds, NULL, NULL, &tv);

  if (-1 == r) { throw std::runtime_error("Error in querying file descriptor"); }
  if (0 == r) { throw std::runtime_error("Querying file descriptor timed out"); }

  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_MMAP;
  if (-1 == ioctl(fd_, VIDIOC_DQBUF, &buf)) { throw std::runtime_error("Failed to deque buffer"); }

  if (buf.index >= num_buffers_.get()) {
    throw std::runtime_error(
        fmt::format("Buf index is {} more than the queue size {}", buf.index, num_buffers_.get()));
  }
}

void V4L2VideoCaptureOp::YUYVToRGBA(const void* yuyv, void* rgba, size_t width, size_t height) {
  auto r_convert = [](int y, int cr) {
    double r = y + (1.4065 * (cr - 128));
    return static_cast<unsigned int>(std::max(0, std::min(255, static_cast<int>(r))));
  };
  auto g_convert = [](int y, int cb, int cr) {
    double g = y - (0.3455 * (cb - 128)) - (0.7169 * (cr - 128));
    return static_cast<unsigned int>(std::max(0, std::min(255, static_cast<int>(g))));
  };
  auto b_convert = [](int y, int cb) {
    double b = y + (1.7790 * (cb - 128));
    return static_cast<unsigned int>(std::max(0, std::min(255, static_cast<int>(b))));
  };

  const unsigned char* yuyv_buf = static_cast<const unsigned char*>(yuyv);
  unsigned char* rgba_buf = static_cast<unsigned char*>(rgba);

  for (unsigned int i = 0, j = 0; i < width * height * 4; i += 8, j += 4) {
    int cb = yuyv_buf[j + 1];
    int cr = yuyv_buf[j + 3];

    // First pixel
    int y = yuyv_buf[j];
    rgba_buf[i] = r_convert(y, cr);
    rgba_buf[i + 1] = g_convert(y, cb, cr);
    rgba_buf[i + 2] = b_convert(y, cb);
    rgba_buf[i + 3] = 255;

    // Second pixel
    y = yuyv_buf[j + 2];
    rgba_buf[i + 4] = r_convert(y, cr);
    rgba_buf[i + 5] = g_convert(y, cb, cr);
    rgba_buf[i + 6] = b_convert(y, cb);
    rgba_buf[i + 7] = 255;
  }
}

// Support for MJPEG format
// Each frame is a JPEG image so use libjpeg to decompress the image and modify it to
// add alpha channel
void V4L2VideoCaptureOp::MJPEGToRGBA(const void* mjpg, void* rgba, size_t width, size_t height) {
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  // Size of image is width * height * 3 (RGB)
  unsigned long jpg_size = width * height * 3;
  int row_stride;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);

  const unsigned char* src_buf =
      const_cast<unsigned char*>(static_cast<const unsigned char*>(mjpg));
  unsigned char* dest_buf = static_cast<unsigned char*>(rgba);
  jpeg_mem_src(&cinfo, src_buf, jpg_size);
  int rc = jpeg_read_header(&cinfo, TRUE);

  if (rc != 1) { throw std::runtime_error("Failed to read jpeg header"); }

  jpeg_start_decompress(&cinfo);

  // Each row has width * 4 pixels (RGBA)
  row_stride = width * 4;

  while (cinfo.output_scanline < cinfo.output_height) {
    unsigned char* buffer_array[1];
    buffer_array[0] = dest_buf + (cinfo.output_scanline) * row_stride;
    // Decompress jpeg image and write it to buffer_arary
    jpeg_read_scanlines(&cinfo, buffer_array, 1);
    unsigned char* buf = buffer_array[0];
    // Modify image to add alpha channel with values set to 255
    // start from the end so we don't overwrite existing values
    for (int i = (int)width * 3 - 1, j = row_stride - 1; i > 0; i -= 3, j -= 4) {
      buf[j] = 255;
      buf[j - 1] = buf[i];
      buf[j - 2] = buf[i - 1];
      buf[j - 3] = buf[i - 2];
    }
  }
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
}

}  // namespace holoscan::ops
