"""
SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""  # noqa: E501

import datetime

from holoscan.conditions import CountCondition, PeriodicCondition
from holoscan.core import Application
from holoscan.operators import PingRxOp, PingTxOp


class MyPingApp(Application):
    def compose(self):
        # Note: Arguments must be positional, not keyword arguments
        tx = PingTxOp(
            self,
            CountCondition(self, 10),
            PeriodicCondition(
                fragment=self,
                recess_period=datetime.timedelta(microseconds=200_000),
                policy="MinTimeBetweenTicks",
                name="noname_periodic_condition",
            ),
            name="tx",
        )
        rx = PingRxOp(self, name="rx")

        self.add_flow(tx, rx)


def main():
    app = MyPingApp()
    app.run()


if __name__ == "__main__":
    main()
