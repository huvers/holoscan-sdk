# Ping Distributed

This example demonstrates a distributed ping application with two operators connected using add_flow().

There are two operators involved in this example:

  1. a transmitter in Fragment 1 (`fragment1`), set to transmit a tensor map containing a single tensor named 'out' on its 'out' port.
  2. a receiver in Fragment 2 (`fragment2`) that prints the received names and shapes of any received tensors to the terminal

The `--gpu` command line argument can be provided to indicate that the tensor should be on the GPU instead of the host (CPU). The user can also override the default tensor shape and data type. Run the application with `-h` or `--help` to see full details of the additional supported arguments.

*Visit the [SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_distributed_app.html) to learn more about distributed applications.*

#### Note on error logged by the application
Note that it is currently expected that this application logs the following error during shutdown

```text
[error] [ucx_context.cpp:466] Connection dropped with status -25 (Connection reset by remote peer)
```

This will be logged by the worker that is running "fragment2" after "fragment1" has sent all messages. It is caused by fragment 1 starting to shutdown after its last message has been sent, resulting in severing of connections from fragment 2 receivers to fragment 1 transmitters.

## C++ Run instructions

Please refer to the [user guide](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_distributed_app.html#building-and-running-a-distributed-application) for instructions on how to run the application in a distributed manner.

### Prerequisites

* **using deb package install or NGC container**:

  ```bash
  # Set the application folder
  APP_DIR=/opt/nvidia/holoscan/examples/ping_distributed/cpp
  ```

* **source (dev container)**:

  ```bash
  ./run launch # optional: append `install` for install tree (default: `build`)

  # Set the application folder
  APP_DIR=./examples/ping_distributed/cpp
  ```

* **source (local env)**:

  ```bash
  # Set the application folder
  APP_DIR=${BUILD_OR_INSTALL_DIR}/examples/ping_distributed/cpp
  ```

### Run the application

```bash
# 1. The following commands will start a driver and one worker in one machine
#    (e.g. IP address `10.2.34.56`) using the port number `10000`,
#    and another worker in another machine.
#    If `--fragments` is not specified, any fragment in the application will be chosen to run.
# 1a. In the first machine (e.g. `10.2.34.56`):
#    (add --gpu to transmit a GPU tensor instead of a host one)
${APP_DIR}/ping_distributed --driver --worker --address 10.2.34.56:10000 --fragments fragment1
# 1b. In the second machine:
${APP_DIR}/ping_distributed --worker --address 10.2.34.56:10000 --fragments fragment2

# 2. The following command will start the distributed app in a single process
#    (add --gpu to transmit a GPU tensor instead of a host one)
${APP_DIR}/ping_distributed
```

Note that for this application "fragment1" sends the video frames and "fragment2" receives them (these fragment names were assigned during the `make_fragment` calls within the `App::compose` method for this app. In this case, "fragment2" has the receiver operator that logs messages to the terminal, so the process that runs that fragment will display the application output. We could omit the `--fragments` arguments altogether if we wanted to let holoscan automatically decide which nodes to run each fragment on. We chose to explicitly specify the fragments here so the user of the application knows which node to expect to see the output on.


## Python Run instructions

Please refer to the [user guide](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_distributed_app.html#building-and-running-a-distributed-application) for instructions on how to run the application in a distributed manner.

### Prerequisites

* **using python wheel**:

  ```bash
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed

  # Set the application folder
  APP_DIR=<APP_DIR>
  ```

* **using deb package or NGC container**:

  ```bash
  # Set the application folder
  APP_DIR=/opt/nvidia/holoscan/examples/ping_distributed/python
  ```

* **source (dev container)**:

  ```bash
  ./run launch # optional: append `install` for install tree (default: `build`)

  # Set the application folder
  APP_DIR=./examples/ping_distributed/python
  ```

* **source (local env)**:

  ```bash
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib

  # Set the application folder
  APP_DIR=${BUILD_OR_INSTALL_DIR}/examples/ping_distributed/python
  ```

### Run the application

```bash
# 1. The following commands will start a driver and one worker in one machine
#    (e.g. IP address `10.2.34.56`) using the port number `10000`,
#    and another worker in another machine.
#    If `--fragments` is not specified, any fragment in the application will be chosen to run.
# 1a. In the first machine (e.g. `10.2.34.56`):
#    (add --gpu to transmit a GPU tensor instead of a host one)
python3 ${APP_DIR}/ping_distributed.py --driver --worker --address 10.2.34.56:10000 --fragments fragment1
# 1b. In the second machine:
python3 ${APP_DIR}/ping_distributed.py --worker --address 10.2.34.56:10000 --fragments fragment2

# 2. The following command will start the distributed app in a single process
#    (add --gpu to transmit a GPU tensor instead of a host one)
python3 ${APP_DIR}/ping_distributed.py
```

Add an additional `--gpu` to the command line to use a GPU tensor instead of a host one.

Note that for this application "fragment1" sends the video frames and "fragment2" receives them (these fragment names were assigned during the `MyPingApp.compose` method for this application. In this case, "fragment2" has the receiver operator that logs messages to the terminal, so the process that runs that fragment will display the application output. We could omit the `--fragments` arguments altogether if we wanted to let holoscan automatically decide which nodes to run each fragment on. We chose to explicitly specify the fragments here so the user of the application knows which node to expect to see the output on.

The `--track` argument can be specified to enable the distributed data flow tracking feature (to measure timings along various paths in the computation graph). Currently this should only be used when the fragments are run on a single node as time synchronization across multiple nodes is not yet automatically handled.
