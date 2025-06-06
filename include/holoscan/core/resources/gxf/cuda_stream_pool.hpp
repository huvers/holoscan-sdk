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

#ifndef HOLOSCAN_CORE_RESOURCES_CUDA_STREAM_POOL_HPP
#define HOLOSCAN_CORE_RESOURCES_CUDA_STREAM_POOL_HPP

#include <string>

#include <gxf/cuda/cuda_stream_pool.hpp>

#include "./allocator.hpp"

namespace holoscan {

/**
 * @brief CUDA stream pool allocator.
 *
 * An allocator that creates a pool of CUDA streams.
 *
 * Internally, the streams created correspond to `nvidia::gxf::CudaStream` objects whose lifetime
 * is managed by the underlying GXF framework.
 *
 * The user can pass this resource to any `make_operator` to allow an operator to control the CUDA
 * stream pool it will use when a stream-related API calls such as
 * `InputContext::receive_cuda_stream` is made from the `Operator::compute` method.
 *
 * ==Parameters==
 *
 * - **dev_id** (int32_t, optional): The CUDA device id specifying which device the memory pool
 * will use (Default: 0).
 * - **stream_flags** (uint32_t, optional): The flags passed to the underlying
 * `cudaStreamCreateWithPriority` CUDA runtime API call when the streams for this pool are created
 * (Default: 0).
 * - **stream_priority_** (int32_t, optional): The priority passed to the underlying
 * `cudaStreamCreateWithPriority` CUDA runtime API call when the streams for this pool are created
 * (Default: 0).
 * - **reserved_size** (uint32_t, optional): The number of streams to initialize the pool with
 * (Default: 1).
 * - **max_size** (uint32_t, optional): The maximum number of streams that can be allocated in the
 * pool. A value of 0 indicates that the size is unlimited (Default: 0). Note that in practice the
 * device hardware will limit the number of possible concurrent kernels and/or memory copy
 * operations to a value defined by `CUDA_DEVICE_MAX_CONNECTIONS`.
 */
class CudaStreamPool : public Allocator {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(CudaStreamPool, Allocator)
  CudaStreamPool() = default;
  CudaStreamPool(int32_t dev_id, uint32_t stream_flags, int32_t stream_priority,
                 uint32_t reserved_size, uint32_t max_size)
      : dev_id_(dev_id),
        stream_flags_(stream_flags),
        stream_priority_(stream_priority),
        reserved_size_(reserved_size),
        max_size_(max_size) {}
  CudaStreamPool(const std::string& name, nvidia::gxf::CudaStreamPool* component);

  const char* gxf_typename() const override { return "nvidia::gxf::CudaStreamPool"; }

  void setup(ComponentSpec& spec) override;

  nvidia::gxf::CudaStreamPool* get() const;

 private:
  Parameter<int32_t> dev_id_;
  Parameter<uint32_t> stream_flags_;
  Parameter<int32_t> stream_priority_;
  Parameter<uint32_t> reserved_size_;
  Parameter<uint32_t> max_size_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_CUDA_STREAM_POOL_HPP */
