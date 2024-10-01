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

#ifndef PYHOLOSCAN_CONDITIONS_DOWNSTREAM_MESSAGE_AFFORDABLE_PYDOC_HPP
#define PYHOLOSCAN_CONDITIONS_DOWNSTREAM_MESSAGE_AFFORDABLE_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace DownstreamMessageAffordableCondition {

PYDOC(DownstreamMessageAffordableCondition, R"doc(
Condition that permits execution when the downstream operator can accept new messages.

Satisfied when the receiver queue of any connected downstream operators has at least a certain
number of elements free. The minimum number of messages that permits the execution of
the entity is specified by `min_size`. It can be used for operators to prevent operators from
sending a message when the downstream operator is not ready to receive it.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment the condition will be associated with
min_size : int
    The minimum number of free slots present in the back buffer.
name : str, optional
    The name of the condition.
)doc")

PYDOC(transmitter, R"doc(
The transmitter associated with the condition.
)doc")

PYDOC(min_size, R"doc(
The minimum number of free slots required for the downstream entity's back
buffer.
)doc")

}  // namespace DownstreamMessageAffordableCondition

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CONDITIONS_DOWNSTREAM_MESSAGE_AFFORDABLE_PYDOC_HPP
