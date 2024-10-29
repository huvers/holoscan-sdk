# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

find_package(ucx 1.17.0 REQUIRED)

install(
  DIRECTORY ${UCX_LIBRARIES}
  DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}/.."
  COMPONENT "holoscan-dependencies"
  FILES_MATCHING PATTERN "*.so*"
)

foreach(ucx_target ucm ucp ucs uct)
  install(
    DIRECTORY ${UCX_INCLUDE_DIRS}/${ucx_target}
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/3rdparty/ucx"
    COMPONENT "holoscan-dependencies"
  )
endforeach()
