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

#ifndef PYHOLOSCAN_CORE_OPERATOR_PYDOC_HPP
#define PYHOLOSCAN_CORE_OPERATOR_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace ParameterFlag {

//  Constructor
PYDOC(ParameterFlag, R"doc(
Enum class for parameter flags.

The following flags are supported:
- `NONE`: The parameter is mendatory and static. It cannot be changed at runtime.
- `OPTIONAL`: The parameter is optional and might not be available at runtime.
- `DYNAMIC`: The parameter is dynamic and might change at runtime.
)doc")

}  // namespace ParameterFlag

namespace OperatorSpec {

//  Constructor
PYDOC(OperatorSpec, R"doc(
Operator specification class.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that this operator belongs to.
)doc")

PYDOC(input, R"doc(
Add an input to the specification.
)doc")

PYDOC(input_kwargs, R"doc(
Add a named input to the specification.

Parameters
----------
name : str
    The name of the input port.
size : int | holoscan.core.IOSpec.IOSize
    The size of the queue for the input port.
    By default, `IOSpec.SIZE_ONE` (== `IOSpec.IOSize(1)`) is used.
    If `IOSpec.ANY_SIZE` is used, it defines multiple receivers internally for the input port.
    Otherwise, the size of the input queue is set to the specified value, and the message available
    condition for the input port is set with `min_size` equal to the same value.

    The following size constants are supported:
    - ``IOSpec.ANY_SIZE``: Any size.
    - ``IOSpec.PRECEDING_COUNT``: Number of preceding connections.
    - ``IOSpec.SIZE_ONE``: The queue size is 1.

    Please refer to the [Holoscan SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_operator.html#receiving-any-number-of-inputs-python)
    to see how to receive any number of inputs in Python.

Notes
-----
The 'size' parameter is used for initializing the queue size of the input port.
The queue size is set by this method or the 'IOSpec.queue_size' property.
If the queue size is set to 'any size' (IOSpec::kAnySize in C++ or IOSpec.ANY_SIZE in Python),
the connector/condition settings will be ignored.
If the queue size is set to other values, the default connector (DoubleBufferReceiver/UcxReceiver)
and condition (MessageAvailableCondition) will use the queue size for initialization
('capacity' for the connector and 'min_size' for the condition) if they are not set.
)doc")

PYDOC(inputs, R"doc(
Return the reference of the input port map.
)doc")

PYDOC(output, R"doc(
Add an output to the specification.
)doc")

PYDOC(output_kwargs, R"doc(
Add a named output to the specification.

Parameters
----------
name : str
    The name of the output port.
)doc")

PYDOC(outputs, R"doc(
Return the reference of the output port map.
)doc")

PYDOC(multi_port_condition, R"doc(
Add a Condition that depends on the status of multiple input ports.

Parameters
----------
port_names : The names of the input ports this condition will apply to.
kwargs : dict or holoscan::ArgList
    Additional arguments to pass to the multi-message condition.
)doc")

PYDOC(multi_port_conditions, R"doc(
Returns a list of multi-message conditions associated with the operator.

Parameters
----------
conditions : list of holoscan.core.MultiMessageConditionInfo
    The list of info structdors for the multi-message conditions associated with the operator.
)doc")

PYDOC(param, R"doc(
Add a parameter to the specification.

Parameters
----------
param : name
    The name of the parameter.
default_value : object
    The default value for the parameter.

Additional Parameters
---------------------
headline : str, optional
    If provided, this is a brief "headline" description for the parameter.
description : str, optional
    If provided, this is a description for the parameter (typically more verbose than the brief
    description provided via `headline`).
kind : str, optional
    In most cases, this keyword should not be specified. If specified, the only valid option is
    currently ``kind="receivers"``, which can be used to create a parameter holding a vector of
    receivers. This effectively creates a multi-receiver input port to which any number of
    operators can be connected.
    Since Holoscan SDK v2.3, users can define a multi-receiver input port using `spec.input()` with
    `size=IOSpec.ANY_SIZE`, instead of using `spec.param()` with `kind="receivers"`. It is now
    recommended to use this new `spec.input`-based approach and the old "receivers" parameter
    approach should be considered deprecated.
flag: holoscan.core.ParameterFlag, optional
    If provided, this is a flag that can be used to control the behavior of the parameter.
    By default, `ParameterFlag.NONE` is used.

    The following flags are supported:
    - `ParameterFlag.NONE`: The parameter is mendatory and static. It cannot be changed at runtime.
    - `ParameterFlag.OPTIONAL`: The parameter is optional and might not be available at runtime.
    - `ParameterFlag.DYNAMIC`: The parameter is dynamic and might change at runtime.

Notes
-----
This method is intended to be called within the `setup` method of an Operator.

In general, for native Python operators, it is not necessary to call `param` to register a
parameter with the class. Instead, one can just directly add parameters to the Python operator
class (e.g. For example, directly assigning ``self.param_name = value`` in __init__.py).

The one case which cannot be implemented without a call to `param` is adding a multi-receiver port
to an operator via a parameter with ``kind="receivers"`` set.
)doc")

PYDOC(description, R"doc(
YAML formatted string describing the operator spec.
)doc")

}  // namespace OperatorSpec

namespace Operator {

//  Constructor
PYDOC(Operator_args_kwargs, R"doc(
Operator class.

Can be initialized with any number of Python positional and keyword arguments.

If a `name` keyword argument is provided, it must be a `str` and will be
used to set the name of the operator.

`Condition` classes will be added to ``self.conditions``, `Resource`
classes will be added to ``self.resources``, and any other arguments will be
cast from a Python argument type to a C++ `Arg` and stored in ``self.args``.
(For details on how the casting is done, see the `py_object_to_arg`
utility). When a Condition or Resource is provided via a kwarg, it's name
will be automatically be updated to the name of the kwarg.

Parameters
----------
fragment : holoscan.core.Fragment
    The `holoscan.core.Fragment` (or `holoscan.core.Application`) to which this Operator will
    belong.
\*args
    Positional arguments.
\*\*kwargs
    Keyword arguments.

Raises
------
RuntimeError
    If `name` kwarg is provided, but is not of `str` type.
    If multiple arguments of type `Fragment` are provided.
    If any other arguments cannot be converted to `Arg` type via `py_object_to_arg`.
)doc")

PYDOC(name, R"doc(
The name of the operator.
)doc")

PYDOC(fragment, R"doc(
The fragment (``holoscan.core.Fragment``) that the operator belongs to.
)doc")

PYDOC(metadata, R"doc(
The metadata dictionary (``holoscan.core.MetadataDictionary``) associated with the operator.
)doc")

PYDOC(is_metadata_enabled, R"doc(
Boolean indicating whether the fragment this operator belongs to has metadata transmission enabled.
)doc")

PYDOC(metadata_policy, R"doc(
The metadata dictionary (``holoscan.core.MetadataPolicy``) associated with the operator.
)doc")

PYDOC(spec, R"doc(
The operator spec (``holoscan.core.OperatorSpec``) associated with the operator.
)doc")

PYDOC(conditions, R"doc(
Conditions associated with the operator.
)doc")

PYDOC(resources, R"doc(
Resources associated with the operator.
)doc")

PYDOC(resource, R"doc(
Resources associated with the operator.

Parameters
----------
name : str
The name of the resource to retrieve

Returns
-------
holoscan.core.Resource or None
    The resource with the given name. If no resource with the given name is found, None is returned.
)doc")

PYDOC(add_arg_Arg, R"doc(
Add an argument to the component.
)doc")

PYDOC(add_arg_ArgList, R"doc(
Add a list of arguments to the component.
)doc")

PYDOC(add_arg_kwargs, R"doc(
Add arguments to the component via Python kwargs.
)doc")

PYDOC(add_arg_condition, R"doc(
)doc")

PYDOC(add_arg_resource, R"doc(
Add a condition or resource to the Operator.

This can be used to add a condition or resource to an operator after it has already been
constructed.

Parameters
----------
arg : holoscan.core.Condition or holoscan.core.Resource
    The condition or resource to add.
)doc")

PYDOC(initialize, R"doc(
Operator initialization method.
)doc")

PYDOC(setup, R"doc(
Operator setup method.
)doc")

PYDOC(start, R"doc(
Operator start method.
)doc")

PYDOC(stop, R"doc(
Operator stop method.
)doc")

PYDOC(compute, R"doc(
Operator compute method. This method defines the primary computation to be
executed by the operator.
)doc")

PYDOC(operator_type, R"doc(
The operator type.

`holoscan.core.Operator.OperatorType` enum representing the type of
the operator. The two types currently implemented are native and GXF.
)doc")

PYDOC(description, R"doc(
YAML formatted string describing the operator.
)doc")

}  // namespace Operator

namespace OperatorType {

PYDOC(OperatorType, R"doc(
Enum class for operator types used by the executor.

- NATIVE: Native operator.
- GXF: GXF operator.
- VIRTUAL: Virtual operator. (for internal use, not intended for use by application authors)
)doc")

}  // namespace OperatorType
}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CORE_OPERATOR_PYDOC_HPP
