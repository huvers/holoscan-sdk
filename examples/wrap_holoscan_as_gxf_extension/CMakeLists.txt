# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_subdirectory(ping_variant_custom_native_res)  # example for Holoscan Resource
add_subdirectory(ping_variant_rx_native_op)
add_subdirectory(ping_variant_tx_native_op)
add_subdirectory(gxf_extension)
add_subdirectory(gxf_app)

if(HOLOSCAN_REGISTER_GXF_EXTENSIONS)
  add_subdirectory(gxf_registry)
endif()

file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

install(
  FILES README.md
  DESTINATION "${app_relative_dest_path}"
  COMPONENT "holoscan-examples"
)
