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

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Copy native operator multithread application
add_custom_target(python_multithread ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/multithread.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "multithread.py"
  BYPRODUCTS "multithread.py"
)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/multithread.py"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT "holoscan-examples"
)

# Install the minimal CMakeLists.txt file
install(FILES CMakeLists.min.txt
  RENAME "CMakeLists.txt"
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Testing
if(HOLOSCAN_BUILD_TESTS)
  add_test(NAME EXAMPLE_PYTHON_MULTITHREAD_TEST
    COMMAND python3 multithread.py --name EXAMPLE_PYTHON_MULTITHREAD_TEST
                                   --output_file multithread_times
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_MULTITHREAD_TEST PROPERTIES
                       PASS_REGULAR_EXPRESSION "sum of received values: 496")

  add_test(NAME EXAMPLE_PYTHON_MULTITHREAD_8_TEST
           COMMAND python3 multithread.py --threads 8 --num_delay_ops 8 --delay 0.25 --delay_step 0.1
                   --name EXAMPLE_PYTHON_MULTITHREAD_8_TEST --output_file multithread_times
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_MULTITHREAD_8_TEST PROPERTIES
                       PASS_REGULAR_EXPRESSION "sum of received values: 28"
                       DEPENDS EXAMPLE_PYTHON_MULTITHREAD_TEST)

  add_test(NAME EXAMPLE_PYTHON_MULTITHREAD_1_TEST
           COMMAND python3 multithread.py --threads 1 --num_delay_ops 8 --delay 0.25 --delay_step 0.1
           --name EXAMPLE_PYTHON_MULTITHREAD_1_TEST --output_file multithread_times
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_MULTITHREAD_1_TEST PROPERTIES
                       PASS_REGULAR_EXPRESSION "sum of received values: 28"
                       DEPENDS EXAMPLE_PYTHON_MULTITHREAD_8_TEST)

  add_test(NAME EXAMPLE_PYTHON_MULTITHREAD_0_TEST
           COMMAND python3 multithread.py --threads 0 --num_delay_ops 8 --delay 0.25 --delay_step 0.1
           --name EXAMPLE_PYTHON_MULTITHREAD_0_TEST --output_file multithread_times
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(EXAMPLE_PYTHON_MULTITHREAD_0_TEST PROPERTIES
                       PASS_REGULAR_EXPRESSION "sum of received values: 28"
                       DEPENDS EXAMPLE_PYTHON_MULTITHREAD_1_TEST)

  # Add a test to check the validity of the timing
  # This test is only valid for dgpu
  execute_process(COMMAND nvidia-smi -L
                  OUTPUT_VARIABLE HOST_GPU)
  if(NOT HOST_GPU MATCHES "nvgpu")
    add_test(NAME EXAMPLE_PYTHON_MULTITHREAD_TIMES_VALIDATION_TEST
            COMMAND python3 ${Holoscan-examples_SOURCE_DIR}/../scripts/ctest_time_comparison.py
                  multithread_times
                  EXAMPLE_PYTHON_MULTITHREAD_TEST LESS EXAMPLE_PYTHON_MULTITHREAD_8_TEST
                  EXAMPLE_PYTHON_MULTITHREAD_8_TEST LESS EXAMPLE_PYTHON_MULTITHREAD_1_TEST
                  EXAMPLE_PYTHON_MULTITHREAD_8_TEST LESS EXAMPLE_PYTHON_MULTITHREAD_0_TEST
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    set_tests_properties(EXAMPLE_PYTHON_MULTITHREAD_TIMES_VALIDATION_TEST PROPERTIES
      DEPENDS EXAMPLE_PYTHON_MULTITHREAD_TEST
      DEPENDS EXAMPLE_PYTHON_MULTITHREAD_8_TEST
      DEPENDS EXAMPLE_PYTHON_MULTITHREAD_1_TEST
      DEPENDS EXAMPLE_PYTHON_MULTITHREAD_0_TEST
      PASS_REGULAR_EXPRESSION "Timing for tests matches expectations"
    )
  endif()

  add_test(NAME EXAMPLE_PYTHON_MULTITHREAD_EVENT_BASED_TEST
    COMMAND python3 multithread.py --event_based
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_MULTITHREAD_TEST PROPERTIES
                       PASS_REGULAR_EXPRESSION "sum of received values: 496")

  add_test(NAME EXAMPLE_PYTHON_MULTITHREAD_TRACK_TEST
          COMMAND python3 multithread.py --track --count 5
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
          )
  set_tests_properties(EXAMPLE_PYTHON_MULTITHREAD_TRACK_TEST PROPERTIES
                       PASS_REGULAR_EXPRESSION "Number of messages: 6"
                       PASS_REGULAR_EXPRESSION "tx->out: 5")
endif()
