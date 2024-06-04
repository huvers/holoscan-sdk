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

#ifndef PYHOLOSCAN_CORE_IO_CONTEXT_PYDOC_HPP
#define PYHOLOSCAN_CORE_IO_CONTEXT_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace Message {

PYDOC(Message, R"doc(
Class representing a message.

A message is a data structure that is used to pass data between operators.
It wraps a ``std::any`` object and provides a type-safe interface to access the data.

This class is used by the `holoscan::gxf::GXFWrapper` to support the Holoscan native operator.
The `holoscan::gxf::GXFWrapper` will hold the object of this class and delegate the message to the
Holoscan native operator.
)doc")

}  // namespace Message

namespace InputContext {

PYDOC(InputContext, R"doc(
Class representing an input context.
)doc")

}  // namespace InputContext

namespace OutputContext {

PYDOC(OutputContext, R"doc(
Class representing an output context.
)doc")

}  // namespace OutputContext

namespace EmitterReceiverRegistry {

PYDOC(EmitterReceiverRegistry, R"doc(
Registry of methods to emit/receive different types.
)doc")

PYDOC(registered_types, R"doc(
List of types with an emitter and/or receiver registered

Returns
-------
names : list of str
	The list of registered emitter/receiver names.
)doc")

}  // namespace EmitterReceiverRegistry

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CORE_IO_CONTEXT_PYDOC_HPP
